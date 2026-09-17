import time

import torch
import torch.nn as nn


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------
    # Problem size
    # -----------------------------------------

    batch = 1
    seq_len = 2
    d_model = 8

    # -----------------------------------------
    # Create input
    # -----------------------------------------

    x = torch.randn(
        batch,
        seq_len,
        d_model,
        device=device
    )

    # -----------------------------------------
    # Create QKV projection
    # -----------------------------------------

    qkv = nn.Linear(
        d_model,
        3 * d_model
    ).to(device)

    # -----------------------------------------
    # Forward
    # -----------------------------------------

    y = qkv(x)

    print("Input shape :", x.shape)
    print("Weight shape:", qkv.weight.shape)
    print("Bias shape  :", qkv.bias.shape)
    print("Output shape:", y.shape)

    # -----------------------------------------
    # Warmup
    # -----------------------------------------

    for _ in range(100):
        y = qkv(x)

    torch.cuda.synchronize()

    # -----------------------------------------
    # Benchmark
    # -----------------------------------------

    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):
        y = qkv(x)

    torch.cuda.synchronize()

    end = time.perf_counter()

    average_ms = (
        (end - start)
        * 1000
        / iterations
    )

    print(
        f"Average QKV projection time: "
        f"{average_ms:.6f} ms"
    )


if __name__ == "__main__":
    main()