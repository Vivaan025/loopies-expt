"""
Phase 4 — Correctness tests for the fused LayerNorm + QKV kernel.

Tests the fused operation against the unfused PyTorch reference:
    ref = F.layer_norm(x, (D,), gamma, beta) @ W_qkv.T + b_qkv

Tests:
  1. FP32 correctness with random weights
  2. FP32 correctness with actual model weights from ExplicitTransformerBlock
  3. FP32 3D input (B, S, D)
  4. BF16 correctness
  5. Microbenchmark: fused vs unfused
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import loop_transformer_kernels
except ImportError:
    print("ERROR: loop_transformer_kernels not installed.")
    print("Run:  cd kernels/ && python setup.py install")
    sys.exit(1)


def unfused_reference(x, gamma, beta, W_qkv, b_qkv, eps=1e-5):
    """The unfused PyTorch reference: LayerNorm then QKV projection."""
    x_norm = F.layer_norm(x, (x.size(-1),), gamma, beta, eps=eps)
    # F.linear does: x_norm @ W_qkv.T + b_qkv
    return F.linear(x_norm, W_qkv, b_qkv)


def test_fp32_random():
    """FP32 with random weights."""

    print("=" * 60)
    print("TEST 1: FP32 fused LN+QKV, random weights")
    print("=" * 60)

    N, D = 1024, 256
    D3 = 3 * D

    x     = torch.randn(N, D, device="cuda", dtype=torch.float32)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32)
    beta  = torch.randn(D, device="cuda", dtype=torch.float32)
    W_qkv = torch.randn(D3, D, device="cuda", dtype=torch.float32)
    b_qkv = torch.randn(D3, device="cuda", dtype=torch.float32)

    our_out = loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
    ref_out = unfused_reference(x, gamma, beta, W_qkv, b_qkv)

    print(f"  Output shape: {our_out.shape}")
    print(f"  Ref shape:    {ref_out.shape}")

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    # FP32 dot product over 256 elements: expect ~1e-4 max diff due to
    # different accumulation order (our kernel vs cuBLAS)
    try:
        torch.testing.assert_close(our_out, ref_out, atol=1e-3, rtol=1e-3)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_fp32_model_weights():
    """FP32 with actual weights from ExplicitTransformerBlock."""

    print("\n" + "=" * 60)
    print("TEST 2: FP32 fused LN+QKV, actual model weights")
    print("=" * 60)

    from attention_v2 import ExplicitTransformerBlock

    torch.manual_seed(42)
    block = ExplicitTransformerBlock(d_model=256, n_heads=8).cuda()
    block.eval()

    N, D = 1024, 256
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)

    # Extract weights
    gamma = block.norm1.weight.detach()
    beta  = block.norm1.bias.detach()
    W_qkv = block.qkv_proj.weight.detach()
    b_qkv = block.qkv_proj.bias.detach()

    our_out = loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
    ref_out = unfused_reference(x, gamma, beta, W_qkv, b_qkv)

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(our_out, ref_out, atol=1e-3, rtol=1e-3)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_fp32_3d():
    """FP32 with (B, S, D) input — verifies batch dim handling."""

    print("\n" + "=" * 60)
    print("TEST 3: FP32 fused LN+QKV, 3D input (B, S, D)")
    print("=" * 60)

    B, S, D = 8, 128, 256
    D3 = 3 * D

    x     = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32)
    beta  = torch.randn(D, device="cuda", dtype=torch.float32)
    W_qkv = torch.randn(D3, D, device="cuda", dtype=torch.float32)
    b_qkv = torch.randn(D3, device="cuda", dtype=torch.float32)

    our_out = loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
    ref_out = unfused_reference(x, gamma, beta, W_qkv, b_qkv)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {our_out.shape}")
    print(f"  Expected:     ({B}, {S}, {D3})")

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        assert our_out.shape == (B, S, D3), f"Shape mismatch: {our_out.shape}"
        torch.testing.assert_close(our_out, ref_out, atol=1e-3, rtol=1e-3)
        print("  PASSED")
        return True
    except (AssertionError, AssertionError) as e:
        print(f"  FAILED: {e}")
        return False


def test_bf16():
    """BF16 — compare fused kernel against BF16 unfused reference."""

    print("\n" + "=" * 60)
    print("TEST 4: BF16 fused LN+QKV")
    print("=" * 60)

    N, D = 1024, 256
    D3 = 3 * D

    x     = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    beta  = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    W_qkv = torch.randn(D3, D, device="cuda", dtype=torch.bfloat16)
    b_qkv = torch.randn(D3, device="cuda", dtype=torch.bfloat16)

    our_out = loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
    ref_out = unfused_reference(x, gamma, beta, W_qkv, b_qkv)

    max_diff = (our_out.float() - ref_out.float()).abs().max().item()
    mean_diff = (our_out.float() - ref_out.float()).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    # BF16 dot product over 256 elements: expect larger diffs
    try:
        torch.testing.assert_close(our_out, ref_out, atol=1.0, rtol=5e-2)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def benchmark():
    """Microbenchmark: fused kernel vs unfused (LayerNorm + F.linear)."""

    print("\n" + "=" * 60)
    print("BENCHMARK: Fused vs Unfused LN+QKV")
    print("=" * 60)

    ITERS = 1000

    for dtype_name, dtype in [("FP32", torch.float32), ("BF16", torch.bfloat16)]:
        N, D = 1024, 256
        D3 = 3 * D

        x     = torch.randn(N, D, device="cuda", dtype=dtype)
        gamma = torch.randn(D, device="cuda", dtype=dtype)
        beta  = torch.randn(D, device="cuda", dtype=dtype)
        W_qkv = torch.randn(D3, D, device="cuda", dtype=dtype)
        b_qkv = torch.randn(D3, device="cuda", dtype=dtype)

        # Warm-up
        for _ in range(50):
            loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
            unfused_reference(x, gamma, beta, W_qkv, b_qkv)
        torch.cuda.synchronize()

        # Fused kernel
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        for _ in range(ITERS):
            loop_transformer_kernels.fused_ln_qkv(x, gamma, beta, W_qkv, b_qkv)
        e1.record()
        torch.cuda.synchronize()
        fused_us = s1.elapsed_time(e1) / ITERS * 1000

        # Unfused reference
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(ITERS):
            unfused_reference(x, gamma, beta, W_qkv, b_qkv)
        e2.record()
        torch.cuda.synchronize()
        unfused_us = s2.elapsed_time(e2) / ITERS * 1000

        print(f"  {dtype_name}: fused={fused_us:.2f} µs, unfused={unfused_us:.2f} µs, "
              f"ratio={unfused_us / fused_us:.2f}x")


if __name__ == "__main__":

    results = []
    results.append(("FP32 random weights",  test_fp32_random()))
    results.append(("FP32 model weights",   test_fp32_model_weights()))
    results.append(("FP32 3D input",        test_fp32_3d()))
    results.append(("BF16",                 test_bf16()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\nAll tests passed! Running benchmark...\n")
        benchmark()
    else:
        print("\nSome tests failed — fix correctness before benchmarking.")
