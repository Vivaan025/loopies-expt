import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

from attention import TransformerBlock


class LoopTransformer(nn.Module):

    def __init__(self, loops=8):
        super().__init__()

        self.block = TransformerBlock()
        self.loops = loops

    def forward(self, x):

        h = x

        for i in range(self.loops):

            with record_function(f"loop_{i + 1}"):
                h = self.block(h)

        return h


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type != "cuda":
        print("CUDA is required.")
        return

    loops = 8

    model = LoopTransformer(loops=loops).to(device)
    model.eval()

    x = torch.randn(
        8,
        128,
        8,
        device=device
    )

    print("Input shape:", x.shape)
    print("Loops:", loops)

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    with torch.no_grad():

        for _ in range(20):
            model(x)

    torch.cuda.synchronize()

    print("\nWarm-up complete.")
    print("Starting profiler...\n")

    # --------------------------------------------------
    # Profiler
    # --------------------------------------------------

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:

        with torch.no_grad():
            model(x)

        torch.cuda.synchronize()

    # --------------------------------------------------
    # Operator summary
    # --------------------------------------------------

    print("\n" + "=" * 100)
    print("PYTORCH OPERATOR SUMMARY")
    print("=" * 100)

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=50
        )
    )

    # --------------------------------------------------
    # ALL PROFILER EVENTS
    # --------------------------------------------------

    print("\n" + "=" * 100)
    print("PROFILER EVENTS")
    print("=" * 100)

    events = prof.events()

    print("Total profiler events:", len(events))

    for i, event in enumerate(events):

        print(
            f"{i:4d} | "
            f"{event.name:60s} | "
            f"CPU total: {event.cpu_time_total:10.3f} us | "
            f"CUDA total: {event.cuda_time_total:10.3f} us"
        )

    # --------------------------------------------------
    # Export trace
    # --------------------------------------------------

    trace_file = "loop_transformer_kernel_trace.json"

    prof.export_chrome_trace(trace_file)

    print("\n" + "=" * 100)
    print("TRACE EXPORTED")
    print("=" * 100)

    print("Trace file:", trace_file)


if __name__ == "__main__":
    main()