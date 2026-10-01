"""
Stage 0 + 1 Script: Reproduce baseline & characterize dynamics.

Stage 0: Verify decomposed forward pass matches library output.
Stage 1: Measure convergence behavior across loop iterations.

Usage:
    python scripts/stage_0_1_dynamics.py [--num-steps 32] [--subset-size 50]

Output:
    results/stage0/    — decomposition verification
    results/stage1/    — convergence analysis data + plots
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_io import (
    load_huginn, get_tokenizer, run_full_decomposed, run_coda,
    run_core_step, verify_decomposition, init_latent_state,
)
from metrics import (
    step_residual, contraction_ratio, cosine_similarity_states,
    step_angle, kl_divergence, top1_flip, oracle_depth, compute_all_stats,
)


def ensure_dirs():
    """Create output directories."""
    os.makedirs("results/stage0", exist_ok=True)
    os.makedirs("results/stage1", exist_ok=True)
    os.makedirs("results/stage1/figures", exist_ok=True)


def run_stage_0(parts, tokenizer, args):
    """Stage 0: Verify decomposed forward pass."""
    print("\n" + "=" * 70)
    print("STAGE 0: Decomposition Verification")
    print("=" * 70)

    # Test inputs
    test_texts = [
        "The capital of France is",
        "In mathematics, the Fibonacci sequence is defined as",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    ]

    results = []
    for text in test_texts:
        print(f"\n  Input: '{text[:50]}...'")
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(parts.device)
        
        result = verify_decomposition(parts, tokens, num_steps=args.num_steps, seed=42)
        result['input'] = text[:50]
        results.append(result)

    # Save results
    with open("results/stage0/verification.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Gate 0 check
    all_pass = all(r.get('pass', False) for r in results)
    print(f"\n{'=' * 70}")
    print(f"GATE 0: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"{'=' * 70}")
    
    if not all_pass:
        print("\nDecomposition does not match library output.")
        print("Check NOTES.md for API differences. Do NOT proceed to Stage 1.")
        print("Common issues:")
        print("  - Random init seed mismatch (Huginn uses random latent state)")
        print("  - Input injection method differs")
        print("  - Layer ordering differs from assumption")
    
    return all_pass


def run_stage_1(parts, tokenizer, args):
    """Stage 1: Characterize convergence dynamics."""
    print("\n" + "=" * 70)
    print("STAGE 1: Convergence Dynamics Analysis")
    print("=" * 70)

    # Prepare evaluation data
    eval_texts = [
        # Factual
        "The theory of general relativity, proposed by Albert Einstein in 1915, describes gravity as the curvature of spacetime caused by mass and energy.",
        "Water molecules consist of two hydrogen atoms and one oxygen atom, giving it the chemical formula H2O.",
        # Reasoning
        "If Alice has 5 apples and gives 2 to Bob, then Bob gives 1 to Charlie, how many apples does each person have?",
        "A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the total distance traveled?",
        # Code
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])",
        # Mathematical
        "The integral of e^x from 0 to 1 equals e - 1, which is approximately 1.718.",
    ]

    N = args.num_steps
    all_per_iter_stats = {
        'residuals_mean': [],    # (N,) mean residual per iter
        'residuals_p95': [],
        'contraction_mean': [],
        'cosine_mean': [],
        'kl_mean': [],
        'kl_p95': [],
        'kl_p99': [],
        'kl_max': [],
        'top1_flip_rate': [],
        'step_angle_mean': [],
    }

    oracle_depths_001 = []
    oracle_depths_01 = []
    oracle_depths_05 = []

    # Seed-to-seed noise floor
    seed_kl_values = []

    for text_idx, text in enumerate(eval_texts):
        print(f"\n  [{text_idx+1}/{len(eval_texts)}] '{text[:60]}...'")
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(parts.device)

        # Run full trajectory with states
        result = run_full_decomposed(
            parts, tokens, num_steps=N, seed=42, return_states=True
        )
        states = result['states']       # [s_0, s_1, ..., s_N]
        logits_ref = result['logits']   # p_N
        prelude_out = result['prelude_out']

        # Compute per-iteration stats
        coda_fn = lambda s: run_coda(parts, s)
        stats = compute_all_stats(states, coda_fn, logits_ref)

        # Aggregate
        for k in range(N):
            r = stats['residuals'][k]
            kl = stats['kl_to_ref'][k]
            flip = stats['top1_flips'][k]
            
            if text_idx == 0:
                all_per_iter_stats['residuals_mean'].append(r.mean().item())
                all_per_iter_stats['residuals_p95'].append(r.quantile(0.95).item())
                all_per_iter_stats['kl_mean'].append(kl.mean().item())
                all_per_iter_stats['kl_p95'].append(kl.quantile(0.95).item())
                all_per_iter_stats['kl_p99'].append(kl.quantile(0.99).item() if kl.numel() > 100 else kl.max().item())
                all_per_iter_stats['kl_max'].append(kl.max().item())
                all_per_iter_stats['top1_flip_rate'].append(flip.float().mean().item())
                all_per_iter_stats['cosine_mean'].append(stats['cosine_sims'][k].mean().item())
            else:
                all_per_iter_stats['residuals_mean'][k] += r.mean().item()
                all_per_iter_stats['residuals_p95'][k] += r.quantile(0.95).item()
                all_per_iter_stats['kl_mean'][k] += kl.mean().item()
                all_per_iter_stats['kl_p95'][k] += kl.quantile(0.95).item()
                all_per_iter_stats['kl_p99'][k] += (kl.quantile(0.99).item() if kl.numel() > 100 else kl.max().item())
                all_per_iter_stats['kl_max'][k] += kl.max().item()
                all_per_iter_stats['top1_flip_rate'][k] += flip.float().mean().item()
                all_per_iter_stats['cosine_mean'][k] += stats['cosine_sims'][k].mean().item()

        if len(stats['contraction_ratios']) > 0:
            for k, rho in enumerate(stats['contraction_ratios']):
                if text_idx == 0:
                    all_per_iter_stats['contraction_mean'].append(rho.mean().item())
                else:
                    all_per_iter_stats['contraction_mean'][k] += rho.mean().item()

        if len(stats['step_angles']) > 0:
            for k, angle in enumerate(stats['step_angles']):
                if text_idx == 0:
                    all_per_iter_stats['step_angle_mean'].append(angle.mean().item())
                else:
                    all_per_iter_stats['step_angle_mean'][k] += angle.mean().item()

        # Oracle depths
        for delta, oracle_list in [(0.001, oracle_depths_001), (0.01, oracle_depths_01), (0.05, oracle_depths_05)]:
            depths = oracle_depth(states, coda_fn, logits_ref, delta=delta)
            oracle_list.append(depths.float().mean().item())

        # Seed-to-seed noise floor (compare seed=42 vs seed=123)
        result_other = run_full_decomposed(parts, tokens, num_steps=N, seed=123)
        kl_seeds = kl_divergence(result_other['logits'], logits_ref)
        seed_kl_values.append(kl_seeds.mean().item())

        # Certificate residual at final state: c(s_N) = ||R(s_N, e) - s_N|| / ||s_N||
        # This measures how close the final state is to a fixed point
        from metrics import certificate_residual
        block_fn = lambda s, e: run_core_step(parts, s, e)
        cert_res = certificate_residual(block_fn, states[-1], prelude_out, per_position=False)
        if text_idx == 0:
            print(f"    Certificate residual c(s_N): {cert_res.item():.6f}")

        # Print progress
        r_2 = stats['residuals'][1].mean().item() if N > 1 else float('nan')
        r_last = stats['residuals'][-1].mean().item()
        print(f"    r_2={r_2:.4f}, r_N={r_last:.4f}, ratio={r_last/r_2:.4f}")
        print(f"    KL(p_N||p_2)={stats['kl_to_ref'][1].mean().item():.4f}, "
              f"KL(p_N||p_N/2)={stats['kl_to_ref'][N//2-1].mean().item():.6f}")

    # Average over texts
    n_texts = len(eval_texts)
    for key in all_per_iter_stats:
        all_per_iter_stats[key] = [v / n_texts for v in all_per_iter_stats[key]]

    # Print summary
    print("\n" + "=" * 70)
    print("STAGE 1: SUMMARY")
    print("=" * 70)

    print(f"\n  Convergence behavior (averaged over {n_texts} sequences):")
    print(f"  {'Iter':>5} {'r_k mean':>10} {'r_k p95':>10} {'cos_sim':>10} {'KL mean':>10} {'flip%':>8}")
    print(f"  {'-'*55}")
    for k in [0, 1, 3, 7, 15, N//2-1, N-1]:
        if k < len(all_per_iter_stats['residuals_mean']):
            print(f"  {k+1:>5} "
                  f"{all_per_iter_stats['residuals_mean'][k]:>10.4f} "
                  f"{all_per_iter_stats['residuals_p95'][k]:>10.4f} "
                  f"{all_per_iter_stats['cosine_mean'][k]:>10.4f} "
                  f"{all_per_iter_stats['kl_mean'][k]:>10.6f} "
                  f"{all_per_iter_stats['top1_flip_rate'][k]*100:>7.1f}%")

    print(f"\n  Oracle depth (mean across sequences):")
    print(f"    delta=0.001: {np.mean(oracle_depths_001):.1f} / {N}")
    print(f"    delta=0.01:  {np.mean(oracle_depths_01):.1f} / {N}")
    print(f"    delta=0.05:  {np.mean(oracle_depths_05):.1f} / {N}")

    print(f"\n  Seed-to-seed noise floor:")
    print(f"    Mean KL(seed=42 || seed=123): {np.mean(seed_kl_values):.6f}")

    # Contraction ratio
    if all_per_iter_stats['contraction_mean']:
        mean_rho = np.mean(all_per_iter_stats['contraction_mean'])
        print(f"\n  Mean contraction ratio rho: {mean_rho:.4f}")
        if mean_rho < 1.0:
            print(f"    → Contractive (rho < 1): Anderson acceleration should work")
        elif mean_rho < 1.05:
            print(f"    → Near-unity (rho ≈ 1): marginal, needs safeguards")
        else:
            print(f"    → Expansive (rho > 1): NOT contractive, Anderson may fail")

    # Gate 1 evaluation
    print(f"\n{'=' * 70}")
    print("GATE 1 EVALUATION:")
    
    oracle_mean_01 = np.mean(oracle_depths_01)
    gate_1a = oracle_mean_01 <= 0.6 * N
    print(f"  (a) Oracle depth at delta=0.01: {oracle_mean_01:.1f} <= {0.6*N:.0f}? "
          f"{'PASS ✓' if gate_1a else 'FAIL ✗'}")
    
    # Check if plain iteration needs >= 6 passes to reach candidate thresholds
    r_at_6 = all_per_iter_stats['residuals_mean'][5] if len(all_per_iter_stats['residuals_mean']) > 5 else float('inf')
    r_threshold = 0.01  # candidate threshold
    gate_1b_iters = next(
        (k+1 for k, r in enumerate(all_per_iter_stats['residuals_mean']) if r < r_threshold),
        N
    )
    gate_1b = gate_1b_iters >= 6
    print(f"  (b) Plain iteration needs {gate_1b_iters} passes to reach r<{r_threshold}: "
          f">= 6? {'PASS ✓' if gate_1b else 'FAIL ✗'}")
    
    overall = gate_1a and gate_1b
    print(f"\n  GATE 1 OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    if not gate_1a:
        print("  → Model needs its depth. Headroom is limited.")
    if not gate_1b:
        print("  → Plain early exit already suffices. Pivot to reporting that finding.")
    print(f"{'=' * 70}")

    # Save all data
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'num_steps': N,
        'num_texts': n_texts,
        'per_iter_stats': all_per_iter_stats,
        'oracle_depths': {
            'delta_0.001': oracle_depths_001,
            'delta_0.01': oracle_depths_01,
            'delta_0.05': oracle_depths_05,
        },
        'seed_kl_noise_floor': seed_kl_values,
        'gate_1a': gate_1a,
        'gate_1b': gate_1b,
        'gate_1_pass': overall,
    }
    
    with open("results/stage1/dynamics_stats.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nResults saved to results/stage1/dynamics_stats.json")
    return overall


def main():
    parser = argparse.ArgumentParser(description="Stage 0+1: Baseline + Dynamics")
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-stage0", action="store_true",
                        help="Skip decomposition verification (if already passed)")
    args = parser.parse_args()

    ensure_dirs()

    # Load model
    parts = load_huginn(args.model, device=args.device)
    tokenizer = get_tokenizer(args.model)

    # Stage 0
    if not args.skip_stage0:
        gate_0 = run_stage_0(parts, tokenizer, args)
        if not gate_0:
            print("\n*** Gate 0 FAILED. Fix decomposition before proceeding. ***")
            print("*** Document findings in NOTES.md. ***")
            sys.exit(1)

    # Stage 1
    gate_1 = run_stage_1(parts, tokenizer, args)
    
    if not gate_1:
        print("\n*** Gate 1 FAILED. Check the gate evaluation above. ***")
        print("*** If (a) failed: model needs its depth. ***")
        print("*** If (b) failed: plain exit suffices; pivot to reporting. ***")
        sys.exit(1)
    
    print("\n✓ Stages 0-1 complete. Proceed to Stage 2 (baselines).")


if __name__ == "__main__":
    main()
