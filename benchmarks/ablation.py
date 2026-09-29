"""
Phase 7 — Ablation study and paper measurements.

Produces the two key tables for the paper:
  1. Ablation: cumulative effect of each optimization
  2. Kernel microbenchmarks: standalone comparisons

All measurements use CUDA events with identical conditions.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import TransformerBlock
from attention_v2 import ExplicitTransformerBlock
from models import LoopTransformer
from models_v2 import LoopTransformerV2

try:
    import loop_transformer_kernels
    HAS_CUSTOM_KERNELS = True
except ImportError:
    HAS_CUSTOM_KERNELS = False
    print("WARNING: Custom kernels not installed. Kernel benchmarks will be skipped.")


# ============================================================
# Timing utilities
# ============================================================

def time_forward(model, x, iterations=200, warmup=20):
    """Time model.forward() using CUDA events. Returns ms/forward."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(iterations):
            model(x)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def time_compiled(model, x, iterations=200, warmup=20):
    """Time torch.compile(model).forward() using CUDA events."""
    model.eval()
    compiled = torch.compile(model)

    with torch.no_grad():
        for _ in range(warmup):
            compiled(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(iterations):
            compiled(x)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def time_cuda_graph(model, x, iterations=200):
    """Time CUDA Graph replay. Returns ms/forward."""
    model.eval()
    compiled = torch.compile(model)

    static_x = x.clone()
    with torch.no_grad():
        for _ in range(3):
            compiled(static_x)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            compiled(static_x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        g.replay()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def time_kernel(fn, iterations=1000, warmup=100):
    """Time a kernel function. Returns µs/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations * 1000  # ms → µs


# ============================================================
# Ablation table
# ============================================================

def run_ablation(B=8, S=128, D=256, loops=8):
    """Ablation study: cumulative effect of each optimization."""

    print("=" * 70)
    print(f"ABLATION TABLE  (B={B}, S={S}, D={D}, loops={loops})")
    print("=" * 70)

    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    results = []

    # ---- Step 0: Eager FP32 ----
    model_fp32 = LoopTransformer(loops=loops).to(device).float()
    x_fp32 = torch.randn(B, S, D, device=device, dtype=torch.float32)

    ms = time_forward(model_fp32, x_fp32)
    results.append(("Eager FP32 (V1)", ms))
    print(f"  [0] Eager FP32 (V1):                    {ms:.4f} ms")

    # ---- Step 1: + BF16 ----
    model_bf16 = model_fp32.bfloat16()
    x_bf16 = x_fp32.bfloat16()

    ms = time_forward(model_bf16, x_bf16)
    results.append(("+ BF16", ms))
    print(f"  [1] + BF16:                             {ms:.4f} ms")

    # ---- Step 2: + torch.compile (V1) ----
    ms = time_compiled(model_bf16, x_bf16)
    results.append(("+ torch.compile (V1)", ms))
    print(f"  [2] + torch.compile (V1):               {ms:.4f} ms")

    # ---- Step 3: + CUDA Graphs (V1) ----
    ms = time_cuda_graph(model_bf16, x_bf16)
    results.append(("+ CUDA Graphs (V1)", ms))
    print(f"  [3] + CUDA Graphs (V1):                 {ms:.4f} ms")

    # ---- Step 4: Explicit QKV (V2), compile + BF16 ----
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_v2 = LoopTransformerV2.from_v1(model_fp32).to(device).bfloat16()

    ms = time_compiled(model_v2, x_bf16)
    results.append(("+ Explicit QKV (V2) + compile", ms))
    print(f"  [4] + Explicit QKV (V2) + compile:      {ms:.4f} ms")

    # ---- Step 5: V2 + CUDA Graphs ----
    ms = time_cuda_graph(model_v2, x_bf16)
    results.append(("+ V2 + CUDA Graphs", ms))
    print(f"  [5] + V2 + CUDA Graphs:                 {ms:.4f} ms")

    # ---- Summary ----
    baseline = results[0][1]
    print("\n  " + "-" * 60)
    print(f"  {'Step':<42} {'ms':>8} {'Speedup':>8}")
    print("  " + "-" * 60)
    for name, ms in results:
        print(f"  {name:<42} {ms:>8.4f} {baseline/ms:>7.2f}×")

    return results


# ============================================================
# Kernel microbenchmarks
# ============================================================

def run_kernel_benchmarks(N=1024, D=256):
    """Standalone kernel timing."""

    if not HAS_CUSTOM_KERNELS:
        print("\nKernel benchmarks skipped (kernels not installed).")
        return

    print("\n" + "=" * 70)
    print(f"KERNEL MICROBENCHMARKS  (N={N}, D={D})")
    print("=" * 70)

    D3 = 3 * D

    for dtype_name, dtype in [("FP32", torch.float32), ("BF16", torch.bfloat16)]:
        print(f"\n  --- {dtype_name} ---")

        x     = torch.randn(N, D, device="cuda", dtype=dtype)
        gamma = torch.randn(D, device="cuda", dtype=dtype)
        beta  = torch.randn(D, device="cuda", dtype=dtype)
        W_qkv = torch.randn(D3, D, device="cuda", dtype=dtype)
        b_qkv = torch.randn(D3, device="cuda", dtype=dtype)

        # PyTorch LayerNorm
        t = time_kernel(lambda: F.layer_norm(x, (D,), gamma, beta))
        print(f"  F.layer_norm:                   {t:>8.2f} µs")

        # Custom LayerNorm
        t_custom_ln = time_kernel(
            lambda: loop_transformer_kernels.layernorm(x, gamma, beta))
        print(f"  Custom LayerNorm:               {t_custom_ln:>8.2f} µs")

        # Unfused: F.layer_norm + F.linear
        t_unfused = time_kernel(
            lambda: F.linear(F.layer_norm(x, (D,), gamma, beta), W_qkv, b_qkv))
        print(f"  F.layer_norm + F.linear:        {t_unfused:>8.2f} µs")

        # Unfused: Custom LN + F.linear
        t_custom_unfused = time_kernel(
            lambda: F.linear(
                loop_transformer_kernels.layernorm(x, gamma, beta), W_qkv, b_qkv))
        print(f"  Custom LN + F.linear:           {t_custom_unfused:>8.2f} µs")

        # Fused LN+QKV
        t_fused = time_kernel(
            lambda: loop_transformer_kernels.fused_ln_qkv(
                x, gamma, beta, W_qkv, b_qkv))
        print(f"  Fused LN+QKV (custom):          {t_fused:>8.2f} µs")


# ============================================================
# Sweep table (different model sizes)
# ============================================================

def run_sweep():
    """Latency sweep across model configurations."""

    print("\n" + "=" * 70)
    print("SWEEP TABLE")
    print("=" * 70)

    configs = [
        # (B, S, D, loops)
        (8, 128, 256, 4),
        (8, 128, 256, 8),
        (8, 128, 256, 16),
        (8, 512, 256, 8),
        (8, 128, 512, 8),
    ]

    print(f"\n  {'B':>3} {'S':>4} {'D':>4} {'L':>3} | "
          f"{'V1 compile':>11} {'V1 graph':>10} {'V2 compile':>11} {'V2 graph':>10}")
    print("  " + "-" * 70)

    for B, S, D, loops in configs:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        device = "cuda"
        x = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)

        # V1
        v1 = LoopTransformer(loops=loops)
        # Adjust d_model if D != 256
        if D != 256:
            v1 = nn.Module()  # Skip non-default sizes for V1
            v1_compile = float('nan')
            v1_graph = float('nan')
        else:
            v1 = v1.to(device).bfloat16()
            v1_compile = time_compiled(v1, x)
            v1_graph = time_cuda_graph(v1, x)

        # V2
        if D != 256:
            v2 = LoopTransformerV2(loops=loops, d_model=D).to(device).bfloat16()
        else:
            v2 = LoopTransformerV2.from_v1(
                LoopTransformer(loops=loops).to(device)).to(device).bfloat16()

        v2_compile = time_compiled(v2, x)
        v2_graph = time_cuda_graph(v2, x)

        print(f"  {B:>3} {S:>4} {D:>4} {loops:>3} | "
              f"{v1_compile:>10.4f}ms {v1_graph:>9.4f}ms "
              f"{v2_compile:>10.4f}ms {v2_graph:>9.4f}ms")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    run_ablation()
    run_kernel_benchmarks()
    run_sweep()

    print("\n" + "=" * 70)
    print("All measurements complete.")
    print("=" * 70)
