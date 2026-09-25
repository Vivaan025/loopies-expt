# Looped Transformer GPU Optimization

## Profiling Phase

This document records the completed PyTorch profiling phase for the shared-weight looped Transformer experiment.

## 1. Research Objective

The project investigates whether repeated execution of a shared Transformer block can be made more efficient on the GPU. The model contains one `TransformerBlock` whose weights are reused for multiple sequential loops.

The current research question is:

> How does repeated execution of a shared Transformer block affect GPU execution efficiency, and which parts of the execution should be optimized at the kernel or system level?

## 2. Model Configuration

| Parameter               | Value                            |
| ----------------------- | -------------------------------- |
| Architecture            | Shared-weight looped Transformer |
| Transformer blocks      | 1 shared block                   |
| Embedding dimension     | 8                                |
| Attention heads         | 2                                |
| Batch size              | 8                                |
| Sequence length         | 128                              |
| Device                  | CUDA GPU                         |
| Profiler workload       | 8 loops                          |
| Loop counts benchmarked | 1, 2, 3, 4, 6, 8                 |

## 3. Completed Experiments

### 3.1 Baseline Scaling

Total forward latency increased as the number of repeated Transformer loops increased. Effective time per loop was approximately 0.65-0.73 ms in the earlier repeated measurements, indicating broadly linear scaling for this workload.

### 3.2 CUDA Graph Experiment

The repeated computation was captured into a CUDA Graph and replayed. Graph replay produced multi-fold latency reductions on the fixed workload, while the maximum numerical difference between normal execution and graph execution was `0.0`.

Measurements varied between runs, particularly for normal execution, so the exact speedup is workload- and environment-dependent. The important finding is that graph replay substantially reduced the execution cost of this small, repetitive workload without changing the computation.

### 3.3 Normal-vs-Graph Execution Gap

The execution gap was calculated as:

```text
execution gap = median normal execution time - median CUDA Graph execution time
```

For the measured workload, the gap was roughly 0.7-0.9 ms per loop in the recorded measurement set. This is described as a normal-vs-graph execution gap rather than pure CPU launch overhead because it may include several forms of execution and orchestration cost.

## 4. Current PyTorch Profiler Findings

An 8-loop forward pass produced 704 profiler events. This number is not equivalent to the number of physical CUDA kernel launches because profiler events include higher-level PyTorch operators and nested or runtime events.

### 4.1 Main Operator-Level Measurements

| Operator                             | Calls | CUDA total | Approx. per call |
| ------------------------------------ | ----: | ---------: | ---------------: |
| `aten::_native_multi_head_attention` |     8 |  11.829 ms |         1.479 ms |
| `aten::linear`                       |    16 |   4.252 ms |         0.266 ms |
| `aten::layer_norm`                   |    16 |   2.604 ms |         0.163 ms |
| `aten::_transform_bias_rescale_qkv`  |     8 |   2.291 ms |         0.286 ms |
| `aten::addmm`                        |    24 |   1.916 ms |         0.080 ms |
| `aten::view`                         |   160 |   1.579 ms |         0.010 ms |
| `aten::transpose`                    |    40 |   1.500 ms |         0.038 ms |
| `aten::empty`                        |    88 |   1.472 ms |         0.017 ms |
| `aten::narrow`                       |    24 |   1.308 ms |         0.055 ms |
| `aten::contiguous`                   |     8 |   0.960 ms |         0.120 ms |
| `aten::reshape`                      |    32 |   0.944 ms |         0.030 ms |
| `aten::clone`                        |     8 |   0.830 ms |         0.104 ms |
| `aten::bmm`                          |    16 |   0.772 ms |         0.048 ms |
| `aten::mm`                           |     8 |   0.530 ms |         0.066 ms |
| `aten::relu`                         |     8 |   0.409 ms |         0.051 ms |

Operator CUDA totals are not all independent. High-level operators such as MultiheadAttention and Linear contain lower-level operations, so their totals must not be added together as if they represented disjoint GPU work.

## 5. What the Trace Reveals

A single Transformer loop is composed of many fine-grained operations rather than one large GPU computation. The attention path repeatedly performs:

- QKV projection
- Tensor splitting and reshaping
- Transposes
- Matrix multiplications
- Softmax
- Output projection
- Memory-layout operations

The repeated pattern includes `view`, `transpose`, `narrow`, `slice`, `as_strided`, `contiguous`, `reshape`, `clone`, and `copy`. These operations indicate that tensor layout and intermediate materialization are meaningful parts of execution.

The model's GEMMs are also small because `d_model` is only 8 and the FFN hidden dimension is 16. Simply replacing PyTorch matrix multiplication with a custom naive CUDA GEMM is not currently justified. PyTorch already relies on optimized CUDA GEMM backends, while this workload may be dominated by fine-grained execution and data-movement costs.

## 6. Current Research Hypothesis

The working hypothesis is that repeated shared-block execution creates significant fine-grained GPU execution overhead. The overhead may be associated with:

- Repeated kernel launches and orchestration
- Tensor-layout transformations
- Intermediate copies
- Normalization
- Many small compute kernels

The CUDA Graph experiment supports this hypothesis because consolidating the repeated execution sequence substantially reduced measured latency. The profiler also shows that the Transformer block is fragmented into repeated operations.

## 7. Why GEMM Is Not the First Optimization Target

The profiler shows matrix multiplications, but the current workload uses very small matrices. Before replacing or tuning GEMMs, the project should determine the actual physical CUDA kernels and their launch and duration characteristics.

This prevents optimizing the wrong layer of the stack and keeps the research focused on the distinctive property of the architecture: repeated execution of the same block.

## 8. Key Takeaway

The current evidence suggests that the research problem is not simply about making one matrix multiplication faster. The looped Transformer repeatedly executes a fine-grained sequence of attention, normalization, GEMM, tensor-layout, and memory operations. CUDA Graph replay demonstrated that consolidating this repetitive execution can materially reduce latency.
