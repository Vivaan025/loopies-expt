"""
Phase 1 — Verify ExplicitTransformerBlock matches original TransformerBlock.

This test proves the refactor is correct BEFORE any CUDA work begins.
Both blocks must produce numerically close outputs given the same weights and input.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import TransformerBlock
from attention_v2 import ExplicitTransformerBlock


def test_single_block():
    """Test that a single ExplicitTransformerBlock matches TransformerBlock."""

    print("=" * 60)
    print("TEST: Single block V1 vs V2")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create original block
    v1 = TransformerBlock(d_model=256, n_heads=8).to(device)
    v1.eval()

    # Create V2 by copying weights from V1
    v2 = ExplicitTransformerBlock.from_original(v1).to(device)
    v2.eval()

    # Same input
    x = torch.randn(8, 128, 256, device=device)

    # Forward
    with torch.no_grad():
        out_v1 = v1(x)
        out_v2 = v2(x)

    # Compare
    max_diff = (out_v1 - out_v2).abs().max().item()
    mean_diff = (out_v1 - out_v2).abs().mean().item()

    print(f"Output V1: shape={out_v1.shape}, mean={out_v1.mean().item():.6f}")
    print(f"Output V2: shape={out_v2.shape}, mean={out_v2.mean().item():.6f}")
    print(f"Max  absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # assert_close uses both atol and rtol
    try:
        torch.testing.assert_close(out_v1, out_v2, atol=1e-4, rtol=1e-4)
        print("PASSED (atol=1e-4, rtol=1e-4)")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return False

    return True


def test_loop_model():
    """Test that LoopTransformer with V2 block matches V1 over 8 loops."""

    print("\n" + "=" * 60)
    print("TEST: LoopTransformer (8 loops) V1 vs V2")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build V1 loop model
    v1_block = TransformerBlock(d_model=256, n_heads=8).to(device)
    v1_block.eval()

    # Build V2 block from same weights
    v2_block = ExplicitTransformerBlock.from_original(v1_block).to(device)
    v2_block.eval()

    # Same input
    x = torch.randn(8, 128, 256, device=device)

    # Run 8 loops through each
    loops = 8

    with torch.no_grad():
        h_v1 = x
        for _ in range(loops):
            h_v1 = v1_block(h_v1)

        h_v2 = x
        for _ in range(loops):
            h_v2 = v2_block(h_v2)

    max_diff = (h_v1 - h_v2).abs().max().item()
    mean_diff = (h_v1 - h_v2).abs().mean().item()

    print(f"After {loops} loops:")
    print(f"Output V1: mean={h_v1.mean().item():.6f}, std={h_v1.std().item():.6f}")
    print(f"Output V2: mean={h_v2.mean().item():.6f}, std={h_v2.std().item():.6f}")
    print(f"Max  absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # After 8 loops, errors accumulate — use looser tolerance
    try:
        torch.testing.assert_close(h_v1, h_v2, atol=1e-3, rtol=1e-3)
        print("PASSED (atol=1e-3, rtol=1e-3)")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return False

    return True


def test_bf16():
    """Test V1 vs V2 in BF16 (what we actually benchmark)."""

    print("\n" + "=" * 60)
    print("TEST: Single block V1 vs V2 (BF16)")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create in FP32, copy, then cast both to BF16
    v1 = TransformerBlock(d_model=256, n_heads=8).to(device)
    v1.eval()

    v2 = ExplicitTransformerBlock.from_original(v1).to(device)
    v2.eval()

    v1 = v1.bfloat16()
    v2 = v2.bfloat16()

    x = torch.randn(8, 128, 256, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out_v1 = v1(x)
        out_v2 = v2(x)

    max_diff = (out_v1 - out_v2).abs().max().item()
    mean_diff = (out_v1 - out_v2).abs().mean().item()

    print(f"Max  absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    # BF16 has less precision — use appropriate tolerance
    try:
        torch.testing.assert_close(out_v1, out_v2, atol=1e-2, rtol=1e-2)
        print("PASSED (atol=1e-2, rtol=1e-2)")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return False

    return True


def test_against_saved_reference():
    """Test V2 output against the Phase 0 saved reference (if it exists)."""

    ref_path = "tests/reference_state_dict.pt"
    ref_fp32_path = "tests/reference_output_fp32.pt"

    if not os.path.exists(ref_path) or not os.path.exists(ref_fp32_path):
        print("\n[SKIP] No saved reference found. Run save_reference.py first.")
        return True

    print("\n" + "=" * 60)
    print("TEST: V2 loop model vs saved reference output")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild V1 from saved state dict
    from models import LoopTransformer

    v1_model = LoopTransformer(loops=8).to(device)
    v1_model.load_state_dict(torch.load(ref_path, weights_only=True))
    v1_model.eval()

    # Build V2 from V1's weights
    v2_block = ExplicitTransformerBlock.from_original(v1_model.block).to(device)
    v2_block.eval()

    # Same seeded input
    x = torch.randn(8, 128, 256, device=device)

    # Run V2 through 8 loops
    with torch.no_grad():
        h = x
        for _ in range(8):
            h = v2_block(h)

    # Load saved reference
    ref_output = torch.load(ref_fp32_path, weights_only=True).to(device)

    max_diff = (h - ref_output).abs().max().item()
    mean_diff = (h - ref_output).abs().mean().item()

    print(f"V2 output mean:  {h.mean().item():.6f}")
    print(f"Reference mean:  {ref_output.mean().item():.6f}")
    print(f"Max  absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(h, ref_output, atol=1e-3, rtol=1e-3)
        print("PASSED (atol=1e-3, rtol=1e-3)")
    except AssertionError as e:
        print(f"FAILED: {e}")
        return False

    return True


if __name__ == "__main__":

    results = []
    results.append(("Single block FP32", test_single_block()))
    results.append(("Loop model FP32",   test_loop_model()))
    results.append(("Single block BF16", test_bf16()))
    results.append(("Vs saved reference", test_against_saved_reference()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in results)
    print(f"\n{'All tests passed!' if all_passed else 'SOME TESTS FAILED'}")
