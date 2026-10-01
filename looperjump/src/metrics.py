"""
metrics.py — Residuals, KL divergence, oracle depth, and damage metrics.

All metrics compare against p_N (reference output at full depth), NOT against
a mathematical fixed point. See plan Section 1 for notation.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def step_residual(s_k: torch.Tensor, s_k_prev: torch.Tensor, per_position: bool = True) -> torch.Tensor:
    """Step residual r_k = ||s_k - s_{k-1}|| / ||s_{k-1}||.
    
    Args:
        s_k: (B, S, D) state after iteration k
        s_k_prev: (B, S, D) state after iteration k-1
        per_position: if True, compute per-position (B, S); else scalar
    Returns:
        residual tensor
    """
    diff = (s_k - s_k_prev).float()
    ref = s_k_prev.float()
    
    if per_position:
        # L2 norm per position
        num = diff.norm(dim=-1)       # (B, S)
        den = ref.norm(dim=-1) + 1e-10
        return num / den
    else:
        return diff.norm() / (ref.norm() + 1e-10)


def contraction_ratio(r_k: torch.Tensor, r_k_prev: torch.Tensor) -> torch.Tensor:
    """Contraction ratio rho_k = r_{k+1} / r_k."""
    return r_k / (r_k_prev + 1e-10)


def certificate_residual(
    block_fn, 
    candidate: torch.Tensor, 
    prelude_out: torch.Tensor,
    per_position: bool = True,
) -> torch.Tensor:
    """Certificate residual c(h) = ||R(h,e) - h|| / ||h||.
    
    This is the self-certifying check: if c(h) is small, h is near a fixed point.
    
    Args:
        block_fn: callable that takes (state, prelude_out) → new_state
        candidate: (B, S, D) candidate state to certify
        prelude_out: (B, S, D) prelude output for input injection
        per_position: per-position or scalar
    """
    with torch.no_grad():
        block_out = block_fn(candidate, prelude_out)
    
    diff = (block_out - candidate).float()
    ref = candidate.float()
    
    if per_position:
        return diff.norm(dim=-1) / (ref.norm(dim=-1) + 1e-10)
    else:
        return diff.norm() / (ref.norm() + 1e-10)


def kl_divergence(logits_method: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor:
    """KL(p_ref || p_method) per position.
    
    Args:
        logits_method: (B, S, V) logits from the method being tested
        logits_ref: (B, S, V) reference logits from full-depth iteration
    Returns:
        kl: (B, S) KL divergence per position
    """
    log_p_method = F.log_softmax(logits_method.float(), dim=-1)
    p_ref = F.softmax(logits_ref.float(), dim=-1)
    
    # KL(p_ref || p_method) = sum p_ref * (log p_ref - log p_method)
    kl = F.kl_div(log_p_method, p_ref, reduction='none').sum(dim=-1)  # (B, S)
    return kl


def top1_flip(logits_method: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor:
    """Top-1 flip: positions where argmax differs from reference.
    
    Returns: (B, S) bool tensor, True = flip (bad)
    """
    return logits_method.argmax(dim=-1) != logits_ref.argmax(dim=-1)


def nll_change(
    logits_method: torch.Tensor,
    logits_ref: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """NLL change per position on teacher-forced text.
    
    NLL_method(t) - NLL_ref(t) for each position t.
    Positive = method is worse than reference.
    
    Args:
        logits_method: (B, S, V)
        logits_ref: (B, S, V)
        labels: (B, S) ground-truth token IDs
    Returns:
        delta_nll: (B, S)
    """
    nll_method = F.cross_entropy(
        logits_method.view(-1, logits_method.size(-1)),
        labels.view(-1),
        reduction='none',
    ).view_as(labels)
    
    nll_ref = F.cross_entropy(
        logits_ref.view(-1, logits_ref.size(-1)),
        labels.view(-1),
        reduction='none',
    ).view_as(labels)
    
    return nll_method - nll_ref


def oracle_depth(
    states: list[torch.Tensor],
    coda_fn,
    logits_ref: torch.Tensor,
    delta: float = 0.01,
) -> torch.Tensor:
    """Oracle depth d*(t, delta): smallest k such that KL(p_N || p_k) < delta.
    
    Args:
        states: list of (B, S, D) states [s_0, s_1, ..., s_N]
        coda_fn: callable(state) → logits
        logits_ref: (B, S, V) reference logits from full-depth (p_N)
        delta: KL threshold
    Returns:
        depths: (B, S) int tensor — oracle depth per position
    """
    B, S = logits_ref.shape[:2]
    N = len(states) - 1  # states[0] is s_0 (random init)
    depths = torch.full((B, S), N, device=logits_ref.device, dtype=torch.long)
    
    for k in range(1, N + 1):
        logits_k = coda_fn(states[k])
        kl_k = kl_divergence(logits_k, logits_ref)  # (B, S)
        
        # Mark positions that haven't reached threshold yet
        not_yet_reached = depths == N
        reached_now = kl_k < delta
        depths[not_yet_reached & reached_now] = k
    
    return depths


def cosine_similarity_states(s_k: torch.Tensor, s_k_prev: torch.Tensor) -> torch.Tensor:
    """Per-position cosine similarity between consecutive states.
    
    Returns: (B, S) values in [-1, 1]. Approaching 1 = convergence.
    """
    return F.cosine_similarity(s_k.float(), s_k_prev.float(), dim=-1)


def step_angle(
    s_k: torch.Tensor,
    s_k_prev: torch.Tensor,
    s_k_prev2: torch.Tensor,
) -> torch.Tensor:
    """Angle between consecutive step directions.
    
    Detects rotation/spiral behavior. Returns angle in radians per position.
    0 = same direction (linear trajectory, easy to predict)
    π = reversal (oscillation)
    """
    d1 = (s_k - s_k_prev).float()          # step k
    d2 = (s_k_prev - s_k_prev2).float()    # step k-1
    
    cos_angle = F.cosine_similarity(d1, d2, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle)  # (B, S)


def compute_all_stats(
    states: list[torch.Tensor],
    coda_fn,
    logits_ref: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> dict:
    """Compute all convergence statistics for a trajectory.
    
    Args:
        states: [s_0, s_1, ..., s_N]
        coda_fn: state → logits
        logits_ref: reference logits (from s_N)
        labels: optional ground-truth tokens for NLL
    Returns:
        dict with per-iteration statistics
    """
    N = len(states) - 1
    
    stats = {
        'residuals': [],           # r_k per iteration
        'contraction_ratios': [],  # rho_k per iteration
        'cosine_sims': [],         # cos(s_k, s_{k-1})
        'step_angles': [],         # angle between consecutive steps
        'kl_to_ref': [],           # KL(p_N || p_k) per iteration
        'top1_flips': [],          # top-1 flip rate per iteration
        'nll_deltas': [],          # NLL change per iteration (if labels provided)
    }
    
    prev_residual = None
    
    for k in range(1, N + 1):
        # Residuals
        r = step_residual(states[k], states[k-1], per_position=True)
        stats['residuals'].append(r)
        
        if prev_residual is not None:
            rho = contraction_ratio(r, prev_residual)
            stats['contraction_ratios'].append(rho)
        prev_residual = r
        
        # Cosine similarity
        stats['cosine_sims'].append(
            cosine_similarity_states(states[k], states[k-1])
        )
        
        # Step angle (needs k >= 2)
        if k >= 2:
            stats['step_angles'].append(
                step_angle(states[k], states[k-1], states[k-2])
            )
        
        # KL to reference
        logits_k = coda_fn(states[k])
        stats['kl_to_ref'].append(
            kl_divergence(logits_k, logits_ref)
        )
        
        # Top-1 flip
        stats['top1_flips'].append(
            top1_flip(logits_k, logits_ref)
        )
        
        # NLL delta
        if labels is not None:
            stats['nll_deltas'].append(
                nll_change(logits_k, logits_ref, labels)
            )
    
    return stats
