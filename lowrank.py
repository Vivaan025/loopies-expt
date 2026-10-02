"""
Phase 8 — Low-rank shared weights.

Instead of storing the shared block's projections as full matrices W,
represent each one directly as a low-rank product:

    W ≈ U V,    U: (d_out, r),  V: (r, d_in),  r << min(d_out, d_in)

This is NOT LoRA. LoRA keeps a full W0 and learns a low-rank *update*
(W0 + AB). Here U, V *are* the weight — there is no full-rank matrix left.
The same U, V are reused on every loop, so any approximation error
E = W - UV is re-applied at every iteration.

    y = x W^T + b   →   y = (x V^T) U^T + b
"""

import copy

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Linear layer whose weight is parameterized as U @ V (rank r)."""

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()

        assert 1 <= rank <= min(in_features, out_features), \
            f"rank {rank} must be in [1, min({in_features}, {out_features})]"

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # x -> V -> (r-dim bottleneck) -> U -> y
        self.V = nn.Linear(in_features, rank, bias=False)
        self.U = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.U(self.V(x))

    @property
    def weight(self):
        """Materialized full weight U @ V (for analysis only)."""
        return self.U.weight @ self.V.weight

    @staticmethod
    def from_linear(linear, rank):
        """Best rank-r approximation of a trained nn.Linear (truncated SVD).

        W = P S Q^T  ->  U = P_r sqrt(S_r),  V = sqrt(S_r) Q_r^T
        By Eckart–Young this minimizes ||W - UV||_F over all rank-r UV.
        """
        W = linear.weight.detach()
        if W.dtype not in (torch.float32, torch.float64):
            W = W.float()   # SVD needs fp32/fp64
        P, S, Qt = torch.linalg.svd(W, full_matrices=False)

        sqrt_s = S[:rank].sqrt()

        lr = LowRankLinear(
            linear.in_features, linear.out_features, rank,
            bias=linear.bias is not None,
        ).to(device=W.device, dtype=W.dtype)

        with torch.no_grad():
            lr.U.weight.copy_(P[:, :rank] * sqrt_s)
            lr.V.weight.copy_(sqrt_s[:, None] * Qt[:rank])
            if linear.bias is not None:
                lr.U.bias.copy_(linear.bias)

        return lr.to(dtype=linear.weight.dtype)


# Projections inside ExplicitTransformerBlock that can be factorized.
TARGETS = {
    "qkv":  ("qkv_proj",),
    "out":  ("out_proj",),
    "ffn1": ("ffn", 0),
    "ffn2": ("ffn", 2),
}
ALL_TARGETS = tuple(TARGETS)


def _get(block, path):
    m = block
    for p in path:
        m = m[p] if isinstance(p, int) else getattr(m, p)
    return m


def _set(block, path, module):
    parent = _get(block, path[:-1])
    if isinstance(path[-1], int):
        parent[path[-1]] = module
    else:
        setattr(parent, path[-1], module)


def compress_block(block, rank, targets=ALL_TARGETS):
    """Return a copy of an ExplicitTransformerBlock with the chosen
    projections replaced by their rank-r SVD factorization.

    `rank` may be an int (same for all targets) or a dict {target: rank}.
    Ranks >= min(d_out, d_in) leave that projection full-rank.
    """
    block = copy.deepcopy(block)

    for name in targets:
        path = TARGETS[name]
        linear = _get(block, path)
        r = rank[name] if isinstance(rank, dict) else rank
        if r >= min(linear.in_features, linear.out_features):
            continue
        _set(block, path, LowRankLinear.from_linear(linear, r))

    return block


def lowrank_block(block, rank, targets=ALL_TARGETS):
    """Replace the chosen projections with freshly initialized rank-r
    factors (for training a low-rank model from scratch)."""
    block = copy.deepcopy(block)

    for name in targets:
        path = TARGETS[name]
        linear = _get(block, path)
        r = rank[name] if isinstance(rank, dict) else rank
        if r >= min(linear.in_features, linear.out_features):
            continue
        _set(block, path, LowRankLinear(
            linear.in_features, linear.out_features, r,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype))

    return block


def count_params(module):
    return sum(p.numel() for p in module.parameters())
