"""
Phase 2 — End-to-end test and benchmark for LoopTransformerV2.

Verifies:
  1. V2 output matches saved FP32 reference
  2. V2 compiled+BF16 latency is within ±5% of V1 baseline
  3. V2 CUDA Graph latency is within ±5% of V1 baseline
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LoopTransformer
from models_v2 import LoopTransformerV2


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("CUDA is required.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Load saved reference
    # --------------------------------------------------

    ref_path = "tests/reference_state_dict.pt"
    ref_fp32_path = "tests/reference_output_fp32.pt"

    if not os.path.exists(ref_path):
        print("ERROR: No saved reference. Run tests/save_reference.py first.")
        return

    # Rebuild V1 from saved weights
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    v1_model = LoopTransformer(loops=8).to(device)
    v1_model.load_state_dict(torch.load(ref_path, weights_only=True))
    v1_model.eval()

    # Build V2 from V1 weights
    v2_model = LoopTransformerV2.from_v1(v1_model).to(device)
    v2_model.eval()

    # --------------------------------------------------
    # Test 1: Output correctness (FP32)
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 1: V2 output vs saved FP32 reference")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(8, 128, 256, device=device)

    with torch.no_grad():
        out_v2 = v2_model(x)

    ref_output = torch.load(ref_fp32_path, weights_only=True).to(device)

    max_diff = (out_v2 - ref_output).abs().max().item()
    mean_diff = (out_v2 - ref_output).abs().mean().item()

    print(f"Max  absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    try:
        torch.testing.assert_close(out_v2, ref_output, atol=1e-3, rtol=1e-3)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")

    # --------------------------------------------------
    # Test 2: Latency comparison (compile + BF16)
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 2: Latency — V1 vs V2 (compile + BF16)")
    print("=" * 60)

    ITERATIONS = 200

    # --- V1 ---
    v1_bf16 = v1_model.bfloat16()
    v1_compiled = torch.compile(v1_bf16)
    x_bf16 = x.bfloat16()

    with torch.no_grad():
        for _ in range(20):
            v1_compiled(x_bf16)
    torch.cuda.synchronize()

    s1 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    s1.record()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            v1_compiled(x_bf16)
    e1.record()
    torch.cuda.synchronize()
    v1_ms = s1.elapsed_time(e1) / ITERATIONS

    # --- V2 ---
    v2_bf16 = v2_model.bfloat16()
    v2_compiled = torch.compile(v2_bf16)

    with torch.no_grad():
        for _ in range(20):
            v2_compiled(x_bf16)
    torch.cuda.synchronize()

    s2 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            v2_compiled(x_bf16)
    e2.record()
    torch.cuda.synchronize()
    v2_ms = s2.elapsed_time(e2) / ITERATIONS

    ratio = v2_ms / v1_ms
    print(f"V1 (compile + BF16): {v1_ms:.4f} ms / forward")
    print(f"V2 (compile + BF16): {v2_ms:.4f} ms / forward")
    print(f"Ratio V2/V1:         {ratio:.3f}")

    if 0.95 <= ratio <= 1.05:
        print("PASSED (within ±5%)")
    else:
        print(f"WARNING: V2 is {(ratio - 1) * 100:+.1f}% vs V1 (outside ±5%)")

    # --------------------------------------------------
    # Test 3: CUDA Graph latency comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 3: CUDA Graph latency — V1 vs V2")
    print("=" * 60)

    def benchmark_graph(compiled_model, input_tensor, iterations=200):
        static_x = input_tensor.clone()
        with torch.no_grad():
            for _ in range(3):
                compiled_model(static_x)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                compiled_model(static_x)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iterations):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iterations

    v1_graph_ms = benchmark_graph(v1_compiled, x_bf16)
    v2_graph_ms = benchmark_graph(v2_compiled, x_bf16)

    ratio_g = v2_graph_ms / v1_graph_ms
    print(f"V1 CUDA Graph: {v1_graph_ms:.4f} ms / forward")
    print(f"V2 CUDA Graph: {v2_graph_ms:.4f} ms / forward")
    print(f"Ratio V2/V1:   {ratio_g:.3f}")

    if 0.95 <= ratio_g <= 1.05:
        print("PASSED (within ±5%)")
    else:
        print(f"WARNING: V2 is {(ratio_g - 1) * 100:+.1f}% vs V1 (outside ±5%)")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"  Correctness (FP32):   max_diff = {max_diff:.2e}")
    print(f"  Compile latency:      V1={v1_ms:.4f}, V2={v2_ms:.4f} ms ({ratio:.3f})")
    print(f"  Graph latency:        V1={v1_graph_ms:.4f}, V2={v2_graph_ms:.4f} ms ({ratio_g:.3f})")


if __name__ == "__main__":
    main()
