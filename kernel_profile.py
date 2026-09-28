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

    # --------------------------------------------------
    # GPU diagnostics
    # --------------------------------------------------
    gpu_name  = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_mem  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated()) / 1024**3
    print(f"GPU          : {gpu_name}")
    print(f"VRAM total   : {total_mem:.1f} GB")
    print(f"VRAM free    : {free_mem:.2f} GB")

    torch.cuda.empty_cache()

    loops = 8

    model = LoopTransformer(loops=loops).to(device, dtype=torch.bfloat16)
    model = torch.compile(model)   # fuse kernels, eliminate launch overhead
    model.eval()

    x = torch.randn(
        8,    # batch size
        128,  # sequence length
        256,  # d_model — must match TransformerBlock
        device=device,
        dtype=torch.bfloat16  # BF16: native on Blackwell, ~2x GEMM throughput
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

    # Capture only this forward
    torch.cuda.profiler.start()

    with torch.no_grad():
        model(x)

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

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
    with_stack=False,
    with_flops=False
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
    print("\n" + "=" * 100)
    print("EVENT TYPES")
    print("=" * 100)

    event_types = {}

    for event in events:

        event_type = str(event.device_type)

        if event_type not in event_types:
            event_types[event_type] = 0

        event_types[event_type] += 1

    for event_type, count in event_types.items():
        print(f"{event_type}: {count}")

    print("Total profiler events:", len(events))

    # for i, event in enumerate(events):

    #     print(
    #         f"{i:4d} | "
    #         f"{event.name:60s} | "
    #         f"CPU total: {event.cpu_time_total:10.3f} us | "
    #         f"CUDA total: {event.cuda_time_total:10.3f} us"
    #     )

    print("\n" + "=" * 100)
    print("DEVICE EVENTS")
    print("=" * 100)

    for i, event in enumerate(events):

        print(
            f"{i:4d} | "
            f"{event.name:60s} | "
            f"Device time: {event.device_time_total:10.3f} us | "
            f"CPU time: {event.cpu_time_total:10.3f} us"
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