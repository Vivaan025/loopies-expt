"""
Phase 3 — Correctness and benchmark tests for the custom LayerNorm kernel.

Tests:
  1. FP32 correctness vs F.layer_norm (ones/zeros gamma/beta)
  2. FP32 correctness vs F.layer_norm (random gamma/beta)
  3. FP32 correctness on 3D input (B, S, D)
  4. BF16 correctness vs F.layer_norm
  5. Microbenchmark: custom kernel vs PyTorch F.layer_norm
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import loop_transformer_kernels
except ImportError:
    print("ERROR: loop_transformer_kernels not installed.")
    print("Run:  cd kernels/ && python setup.py install")
    sys.exit(1)


def test_fp32_identity_params():
    """Test with gamma=1, beta=0 (pure normalization, no affine scaling)."""

    print("=" * 60)
    print("TEST 1: FP32 LayerNorm, gamma=1, beta=0")
    print("=" * 60)

    N, D = 1024, 256
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    gamma = torch.ones(D, device="cuda", dtype=torch.float32)
    beta = torch.zeros(D, device="cuda", dtype=torch.float32)

    our_out = loop_transformer_kernels.layernorm(x, gamma, beta, eps=1e-5)
    ref_out = F.layer_norm(x, (D,), gamma, beta, eps=1e-5)

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_fp32_random_params():
    """Test with random gamma and beta."""

    print("\n" + "=" * 60)
    print("TEST 2: FP32 LayerNorm, random gamma/beta")
    print("=" * 60)

    N, D = 1024, 256
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32)
    beta = torch.randn(D, device="cuda", dtype=torch.float32)

    our_out = loop_transformer_kernels.layernorm(x, gamma, beta, eps=1e-5)
    ref_out = F.layer_norm(x, (D,), gamma, beta, eps=1e-5)

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_fp32_3d_input():
    """Test with (B, S, D) input — verifies shape flattening/unflattening."""

    print("\n" + "=" * 60)
    print("TEST 3: FP32 LayerNorm, 3D input (B, S, D)")
    print("=" * 60)

    B, S, D = 8, 128, 256
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    gamma = torch.randn(D, device="cuda", dtype=torch.float32)
    beta = torch.randn(D, device="cuda", dtype=torch.float32)

    our_out = loop_transformer_kernels.layernorm(x, gamma, beta, eps=1e-5)
    ref_out = F.layer_norm(x, (D,), gamma, beta, eps=1e-5)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {our_out.shape}")

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        assert our_out.shape == x.shape, f"Shape mismatch: {our_out.shape} vs {x.shape}"
        torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)
        print("  PASSED")
        return True
    except (AssertionError, AssertionError) as e:
        print(f"  FAILED: {e}")
        return False


def test_bf16():
    """Test BF16 kernel — looser tolerance due to reduced precision."""

    print("\n" + "=" * 60)
    print("TEST 4: BF16 LayerNorm")
    print("=" * 60)

    N, D = 1024, 256
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(D, device="cuda", dtype=torch.bfloat16)

    our_out = loop_transformer_kernels.layernorm(x, gamma, beta, eps=1e-5)
    ref_out = F.layer_norm(x, (D,), gamma, beta, eps=1e-5)

    max_diff = (our_out.float() - ref_out.float()).abs().max().item()
    mean_diff = (our_out.float() - ref_out.float()).abs().mean().item()

    print(f"  Max  diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(our_out, ref_out, atol=1e-2, rtol=1e-2)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def benchmark():
    """Microbenchmark: custom kernel vs F.layer_norm."""

    print("\n" + "=" * 60)
    print("BENCHMARK: Custom vs PyTorch LayerNorm")
    print("=" * 60)

    ITERS = 1000

    for dtype_name, dtype in [("FP32", torch.float32), ("BF16", torch.bfloat16)]:
        N, D = 1024, 256
        x = torch.randn(N, D, device="cuda", dtype=dtype)
        gamma = torch.ones(D, device="cuda", dtype=dtype)
        beta = torch.zeros(D, device="cuda", dtype=dtype)

        # Warm-up
        for _ in range(50):
            loop_transformer_kernels.layernorm(x, gamma, beta)
            F.layer_norm(x, (D,), gamma, beta)
        torch.cuda.synchronize()

        # Custom kernel
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        for _ in range(ITERS):
            loop_transformer_kernels.layernorm(x, gamma, beta)
        e1.record()
        torch.cuda.synchronize()
        custom_us = s1.elapsed_time(e1) / ITERS * 1000  # ms → µs

        # PyTorch reference
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(ITERS):
            F.layer_norm(x, (D,), gamma, beta)
        e2.record()
        torch.cuda.synchronize()
        pytorch_us = s2.elapsed_time(e2) / ITERS * 1000

        ratio = pytorch_us / custom_us
        print(f"  {dtype_name}: custom={custom_us:.2f} µs, PyTorch={pytorch_us:.2f} µs, ratio={ratio:.2f}x")


if __name__ == "__main__":

    results = []
    results.append(("FP32 identity params", test_fp32_identity_params()))
    results.append(("FP32 random params",   test_fp32_random_params()))
    results.append(("FP32 3D input",        test_fp32_3d_input()))
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
