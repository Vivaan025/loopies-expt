"""
Phase 0 — Save reference baseline outputs and latencies.

Run this ONCE on the 5090 to lock in the baseline before any changes.
All future phases compare against these saved outputs.
"""

import os
import sys

import torch
import torch.nn as nn

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LoopTransformer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("CUDA is required.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Deterministic seeds
    # --------------------------------------------------

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # --------------------------------------------------
    # Build model and input (FP32 first)
    # --------------------------------------------------

    model = LoopTransformer(loops=8).to(device)
    model.eval()

    x = torch.randn(8, 128, 256, device=device)

    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # --------------------------------------------------
    # Save state dict
    # --------------------------------------------------

    os.makedirs("tests", exist_ok=True)

    torch.save(model.state_dict(), "tests/reference_state_dict.pt")
    print("Saved: tests/reference_state_dict.pt")

    # --------------------------------------------------
    # Save FP32 reference output (no compile, eager)
    # --------------------------------------------------

    with torch.no_grad():
        out_fp32 = model(x)

    torch.save(out_fp32, "tests/reference_output_fp32.pt")
    print(
        f"Reference FP32: shape={out_fp32.shape}, dtype={out_fp32.dtype}, "
        f"mean={out_fp32.mean().item():.6f}, std={out_fp32.std().item():.6f}"
    )
    print("Saved: tests/reference_output_fp32.pt")

    # --------------------------------------------------
    # Save BF16 reference output (no compile, eager)
    # --------------------------------------------------

    model_bf16 = model.bfloat16()
    x_bf16 = x.bfloat16()

    with torch.no_grad():
        out_bf16 = model_bf16(x_bf16)

    torch.save(out_bf16, "tests/reference_output_bf16.pt")
    print(
        f"Reference BF16: shape={out_bf16.shape}, dtype={out_bf16.dtype}, "
        f"mean={out_bf16.mean().item():.6f}, std={out_bf16.std().item():.6f}"
    )
    print("Saved: tests/reference_output_bf16.pt")

    # --------------------------------------------------
    # Baseline latency: torch.compile + BF16
    # --------------------------------------------------

    print("\n--- Latency Benchmarks ---")

    model_compiled = torch.compile(model_bf16)

    # Warm-up (compile trace happens here)
    with torch.no_grad():
        for _ in range(20):
            model_compiled(x_bf16)
    torch.cuda.synchronize()

    ITERATIONS = 200

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            model_compiled(x_bf16)
    end.record()
    torch.cuda.synchronize()

    baseline_ms = start.elapsed_time(end) / ITERATIONS
    print(f"Baseline (compile + BF16) : {baseline_ms:.4f} ms / forward")

    # --------------------------------------------------
    # CUDA Graph latency
    # --------------------------------------------------

    static_x = x_bf16.clone()

    # Warm graph capture path
    with torch.no_grad():
        for _ in range(3):
            model_compiled(static_x)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_out = model_compiled(static_x)
    torch.cuda.synchronize()

    # Benchmark replay
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)

    start2.record()
    for _ in range(ITERATIONS):
        g.replay()
    end2.record()
    torch.cuda.synchronize()

    graph_ms = start2.elapsed_time(end2) / ITERATIONS
    print(f"CUDA Graph                : {graph_ms:.4f} ms / forward")
    print(f"Speedup (Graph/Baseline)  : {baseline_ms / graph_ms:.2f}x")

    # --------------------------------------------------
    # Verification: reload and check
    # --------------------------------------------------

    print("\n--- Verification ---")
    loaded_fp32 = torch.load("tests/reference_output_fp32.pt", weights_only=True)
    loaded_bf16 = torch.load("tests/reference_output_bf16.pt", weights_only=True)
    print(f"Reloaded FP32: shape={loaded_fp32.shape}, mean={loaded_fp32.mean().item():.6f}")
    print(f"Reloaded BF16: shape={loaded_bf16.shape}, mean={loaded_bf16.mean().item():.6f}")

    print("\nPhase 0 complete.")


if __name__ == "__main__":
    main()
