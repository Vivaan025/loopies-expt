import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import statistics
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
    loop_count = [1,2,3,4,6,8]

    # model = LoopTransformer(
    #     loops=loops
    # ).to(device)

    # model.eval()

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device
    )

    print("Input shape :", x.shape)

    benchmark_trials = 5

    # profile_loops = 8
    loop_counts = [1, 2, 3, 4, 6, 8]
    for loops in loop_counts:

        print(f"\n========== {loops} LOOPS ==========")

        profile_model = LoopTransformer(
            loops=loops
        ).to(device)

        profile_model.eval()

        if device.type == "cuda":
            graph = torch.cuda.CUDAGraph()
            static_x = x.clone()

            with torch.no_grad():
                for _ in range(3):
                    profile_model(static_x)

            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_output = profile_model(static_x)

            graph.replay()

            with torch.no_grad():
                normal_output = profile_model(static_x)

            graph_output = static_output.clone()

            torch.cuda.synchronize()

            difference = torch.max(
                torch.abs(normal_output - graph_output)
            )

            print(f"Max difference between normal and graph output: {difference.item()}")

        normal_times = []
        graph_times = []


        for trial in range(benchmark_trials):
            iterations = 1000

        
            # -------------------------
            # Normal execution
            # -------------------------
            
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(iterations):
                    profile_model(static_x)
            
            torch.cuda.synchronize()
            
            normal_time = (time.perf_counter() - start) * 1000 / iterations
        
        
            # -------------------------
            # CUDA Graph execution
            # -------------------------
            
            torch.cuda.synchronize()
        
            start = time.perf_counter()
            
            for _ in range(iterations):
                graph.replay()
            
            torch.cuda.synchronize()
            
            graph_time = (time.perf_counter() - start) * 1000 / iterations

            normal_times.append(normal_time)
            graph_times.append(graph_time)
            
            
            print("\n========== CUDA GRAPH BENCHMARK ==========")
            print(f"Trial {trial + 1}/{benchmark_trials}")
            print(f"Iterations       : {iterations}")
            print(f"Normal execution : {normal_time:.6f} ms")
            print(f"CUDA Graph       : {graph_time:.6f} ms")
            print(f"Speedup          : {normal_time / graph_time:.2f}x")
            print(f"Reduction        : {(1 - graph_time / normal_time) * 100:.2f}%")

        median_normal = statistics.median(normal_times)
        median_graph = statistics.median(graph_times)

        speedup = median_normal / median_graph
        reduction = (1 - median_graph / median_normal) * 100

        overhead_gap = median_normal - median_graph

        gap_per_loop = overhead_gap / loops


        print("\n========== MEDIAN RESULT ==========")
        print(f"Normal execution : {median_normal:.6f} ms")
        print(f"CUDA Graph       : {median_graph:.6f} ms")
        print(f"Speedup          : {speedup:.2f}x")
        print(f"Reduction        : {reduction:.2f}%")
        print(f"Execution gap    : {overhead_gap:.6f} ms")
        print(f"Gap per loop     : {gap_per_loop:.6f} ms")

        if loops == 8:
            print("\n========== 8-LOOP PROFILER ==========")

            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                record_shapes=True,
                profile_memory=True
            ) as prof:

                with torch.no_grad():
                    profile_model(static_x)

                torch.cuda.synchronize()

            print(
                prof.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=30
                )
            )


    # profile_model = LoopTransformer(loops=loop_counts).to(device)

    # profile_model.eval()

    with torch.no_grad():
        profile_model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        with torch.no_grad():
            profile_model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

    # print("\n========== PROFILE ==========\n")

    # print(
    #     prof.key_averages().table(
    #         sort_by="cuda_time_total",
    #         row_limit=30
    #     )
    # )


    # print("Loops       :", loops)

    # for loops in loop_count:
    #     print(f"Loops: {loops}")
    #     model = LoopTransformer(loops=loops).to(device)

    #     times = []
        
    #     for _ in range(5):
    #         average_ms = (benchmark_model(model, x, iterations=200))
    #         times.append(average_ms)

    #     median_ms = statistics.median(times)
    #     per_loop_ms = median_ms / loops
        # print(f"Average time: {average_ms:.4f} ms")
        # print(f"Median time: {median_ms:.4f} ms")
        # print(f"Time per loop: {per_loop_ms:.4f} ms")

    # Warm-up
    # with torch.no_grad():
    #     for _ in range(20):
    #         model(x)

    # if device.type == "cuda":
    #     torch.cuda.synchronize()

    # # Profile
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA
    #     ],
    #     record_shapes=True,
    #     profile_memory=True
    # ) as prof:

    #     with torch.no_grad():
    #         model(x)

    #     if device.type == "cuda":
    #         torch.cuda.synchronize()

    # print("\n========== PROFILE ==========\n")

    # print(
    #     prof.key_averages().table(
    #         sort_by="cuda_time_total",
    #         row_limit=30
    #     )
    # )


if __name__ == "__main__":
    main()