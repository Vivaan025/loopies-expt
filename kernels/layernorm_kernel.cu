/*
 * Phase 3 — Standalone LayerNorm CUDA kernel.
 *
 * Launch: one block per row (per token).
 * Block:  D threads (one thread per feature dimension).
 *
 * For D=256, that's 8 warps of 32 threads.
 *
 * Algorithm per block:
 *   1. Load x[row, i]
 *   2. Warp reduction → mean
 *   3. Warp reduction → variance
 *   4. Normalize: (x - mean) / sqrt(var + eps)
 *   5. Affine: gamma * x_hat + beta
 *   6. Store output
 *
 * Accumulations done in FP32 regardless of input dtype.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>


// ============================================================
// Warp-level reduction: sum 32 values using shuffle
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Full warp mask
    const unsigned int mask = 0xFFFFFFFF;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;  // only lane 0 of each warp has the correct sum
}


// ============================================================
// Block-level reduction: reduce across all warps
// ============================================================
//
// Pattern:
//   1. Each warp internally reduces its 32 values → warp sum
//   2. Lane 0 of each warp writes its sum to shared memory
//   3. First warp loads those partial sums and reduces them
//   4. Result is broadcast to all threads via shared memory
//
__device__ float block_reduce_sum(float val, float* smem) {

    const int lane   = threadIdx.x % 32;    // lane within warp
    const int warp   = threadIdx.x / 32;    // which warp
    const int nwarps = blockDim.x / 32;     // total warps in block

    // Step 1: intra-warp reduction
    val = warp_reduce_sum(val);

    // Step 2: warp leaders write to shared memory
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    // Step 3: first warp reduces the warp sums
    // Only the first nwarps threads participate
    float warp_val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
        warp_val = warp_reduce_sum(warp_val);
    }

    // Step 4: broadcast result to all threads
    if (threadIdx.x == 0) {
        smem[0] = warp_val;
    }
    __syncthreads();

    return smem[0];
}


// ============================================================
// LayerNorm kernel — FP32
// ============================================================
__global__ void layernorm_fp32_kernel(
    const float* __restrict__ X,       // input:  (N, D)
    const float* __restrict__ gamma,   // weight: (D,)
    const float* __restrict__ beta,    // bias:   (D,)
          float* __restrict__ Y,       // output: (N, D)
    int N,
    int D,
    float eps
) {
    // Shared memory for cross-warp reductions
    // Need at most (blockDim.x / 32) floats = 8 for D=256
    extern __shared__ float smem[];

    const int row = blockIdx.x;    // which row (token)
    const int i   = threadIdx.x;   // which feature

    if (row >= N || i >= D) return;

    // Load input value
    float x_val = X[row * D + i];

    // --- Compute mean ---
    float sum = block_reduce_sum(x_val, smem);
    float mean = sum / (float)D;

    // --- Compute variance ---
    float diff = x_val - mean;
    float var_sum = block_reduce_sum(diff * diff, smem);
    float var = var_sum / (float)D;

    // --- Normalize ---
    float x_hat = diff / sqrtf(var + eps);

    // --- Affine transform ---
    float y_val = gamma[i] * x_hat + beta[i];

    // --- Store ---
    Y[row * D + i] = y_val;
}


// ============================================================
// LayerNorm kernel — BF16 input/output, FP32 accumulation
// ============================================================
__global__ void layernorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
          __nv_bfloat16* __restrict__ Y,
    int N,
    int D,
    float eps
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    const int i   = threadIdx.x;

    if (row >= N || i >= D) return;

    // Load and convert to FP32 for accumulation
    float x_val = __bfloat162float(X[row * D + i]);

    // --- Compute mean ---
    float sum = block_reduce_sum(x_val, smem);
    float mean = sum / (float)D;

    // --- Compute variance ---
    float diff = x_val - mean;
    float var_sum = block_reduce_sum(diff * diff, smem);
    float var = var_sum / (float)D;

    // --- Normalize ---
    float x_hat = diff / sqrtf(var + eps);

    // --- Affine transform (load gamma/beta as FP32) ---
    float g = __bfloat162float(gamma[i]);
    float b = __bfloat162float(beta[i]);
    float y_val = g * x_hat + b;

    // --- Store as BF16 ---
    Y[row * D + i] = __float2bfloat16(y_val);
}


// ============================================================
// C-linkage launcher functions (called from binding.cpp)
// ============================================================

extern "C" {

void launch_layernorm_fp32(
    const float* X, const float* gamma, const float* beta, float* Y,
    int N, int D, float eps, cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    // Shared memory: one float per warp
    int smem_size = (D / 32) * sizeof(float);
    layernorm_fp32_kernel<<<grid, block, smem_size, stream>>>(
        X, gamma, beta, Y, N, D, eps
    );
}

void launch_layernorm_bf16(
    const __nv_bfloat16* X, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_bfloat16* Y,
    int N, int D, float eps, cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    int smem_size = (D / 32) * sizeof(float);
    layernorm_bf16_kernel<<<grid, block, smem_size, stream>>>(
        X, gamma, beta, Y, N, D, eps
    );
}

}  // extern "C"
