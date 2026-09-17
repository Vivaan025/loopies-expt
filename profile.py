import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
from attention import TransformerBlock


class LoopTransformer(nn.Module):

    def __init__(self, loops=3):
        super().__init__()

        # One shared Transformer block
        self.block = TransformerBlock()

        self.loops = loops

    def forward(self, x):

        h = x

        for i in range(self.loops):

            # This gives the profiler a name for each loop.
            with record_function(f"loop_{i + 1}"):
                h = self.block(h)

        return h

def benchmark_model(model, x, iterations=100):
        model.eval()

        with torch.no_grad():
            for _ in range(20):
                model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter()

        total_time = end - start
        average_time = total_time * 1000 / iterations

        return average_time



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Start with a more useful size than [1, 2, 8].
    batch_size = 8
    sequence_length = 128
    d_model = 8
    loops = 3

    model = LoopTransformer(
        loops=loops
    ).to(device)

    model.eval()

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device
    )

    print("Input shape :", x.shape)
    print("Loops       :", loops)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        with torch.no_grad():
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

    print("\n========== PROFILE ==========\n")

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30
        )
    )


if __name__ == "__main__":
    main()