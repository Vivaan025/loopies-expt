# import torch
# from torch import nn
# from attention import TransformerBlock

# class LoopTransformer(nn.Module):
#     def __init__(self, loops=3):
#         super().__init__()
        
#         self.block = TransformerBlock()

#         self.loops = loops

#     def forward(self, x):

#         h = x

#         for i in range(self.loops):
#             previous_h = h.detach().clone()
#             # print(f"\nLoop {i + 1}")
#             # print(h)
#             # print("Loop", i + 1, "block object:", id(self.block))
#             h = self.block(h)

#             difference = (h - previous_h).abs().mean().item()

#             # print(
#                 # f"Loop {i + 1}: "
#                 # f"mean absolute change = {difference:.6f}"
#             # )


#         # print(
#         #     f"Loop {i+1}: "
#         #     f"mean={h.mean().item():.4f}, "
#         #     f"norm={h.norm().item():.4f}"
#         # )
        

#         return h

# # model = LoopTransformer(loops=3)
# # print("\nQKV weight shape:")
# # print(model.block.attention.in_proj_weight.shape)

# # print("\nQKV bias shape:")
# # print(model.block.attention.in_proj_bias.shape)

# # print("\nOutput projection weight shape:")
# # print(model.block.attention.out_proj.bias.shape)
# # print("Block object:", id(model.block))
# # print("\nBefore training:")
# # print(model.block.attention.in_proj_weight.grad)

# # x = torch.randn(1, 2, 8)

# # output = model(x)

# # print("input shape :", x.shape)
# # print("output shape:", output.shape)

# # print("Number of parameters:", sum(p.numel() for p in model.parameters()))

import time

import torch
import torch.nn as nn

from attention import TransformerBlock


class LoopTransformer(nn.Module):

    def __init__(self, loops=3):
        super().__init__()

        # ONE shared Transformer block
        self.block = TransformerBlock()

        # Number of times the block is reused
        self.loops = loops

    def forward(self, x):

        h = x

        for _ in range(self.loops):
            h = self.block(h)

        return h


def benchmark(model, x, iterations=100, warmup=20):

    model.eval()

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------
    with torch.no_grad():

        for _ in range(warmup):
            model(x)

    # Make sure GPU has finished the warm-up work
    if x.is_cuda:
        torch.cuda.synchronize()

    # --------------------------------------------------
    # Start timing
    # --------------------------------------------------
    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):
            output = model(x)

    # CUDA operations are asynchronous.
    # Wait for GPU before stopping timer.
    if x.is_cuda:
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start

    average_time_ms = (total_time * 1000) / iterations

    return output, average_time_ms


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loops = 3

    batch_size = 1
    sequence_length = 2
    d_model = 8

    print("Device:", device)
    print("Loops:", loops)

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = LoopTransformer(
        loops=loops
    ).to(device)

    # --------------------------------------------------
    # Create input
    # --------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device
    )

    print("Input shape :", x.shape)

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters  :", parameter_count)

    # --------------------------------------------------
    # Verify shared block
    # --------------------------------------------------

    print("Block object:", id(model.block))

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    output, average_time_ms = benchmark(
        model,
        x,
        iterations=100,
        warmup=20
    )

    print("Output shape:", output.shape)

    print(
        f"Average forward time: "
        f"{average_time_ms:.4f} ms"
    )

    # --------------------------------------------------
    # GPU memory
    # --------------------------------------------------

    if device.type == "cuda":

        allocated_mb = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved_mb = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        print(
            f"GPU memory allocated: "
            f"{allocated_mb:.2f} MB"
        )

        print(
            f"GPU memory reserved: "
            f"{reserved_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()