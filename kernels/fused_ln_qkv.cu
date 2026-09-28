/*
 * Phase 5 — Optimized Fused LayerNorm + QKV Projection kernel.
 *
 * Improvements over Phase 4 (correctness prototype):
 *   - Warp-parallel GEMV: each warp cooperatively computes one dot product
 *     with coalesced global memory reads (vs Phase 4's strided reads)
 *   - 32× reduction in memory amplification for weight reads
 *   - No __syncthreads in the GEMV phase (only warp-level shuffles)
 *
 * Design:
 *   - One block per row (token), 256 threads = 8 warps
 *   - Phase 1: LayerNorm with warp/block reduction → x_norm in shared memory
 *   - Phase 2: Warp-parallel GEMV
 *     - Each warp handles one output column per iteration
 *     - 8 columns processed simultaneously (one per warp)
 *     - 768 / 8 = 96 outer iterations
 *     - Per iteration: each thread loads D/32 = 8 weight values (coalesced)
 *     - Warp shuffle reduction → lane 0 has the full dot product
 *
 * Memory access pattern:
 *   Phase 4: threads in warp read W[col0*D+k], W[col1*D+k], ... (stride D)
 *   Phase 5: threads in warp read W[col*D+0], W[col*D+1], ... (contiguous)
 *
 * Input:   X       (N, D)     — pre-norm activations
 *          gamma   (D,)       — LayerNorm weight
 *          beta    (D,)       — LayerNorm bias
 *          W_qkv   (3D, D)    — packed QKV projection weight (row-major)
 *          b_qkv   (3D,)      — packed QKV projection bias
 * Output:  OUT     (N, 3D)    — packed Q, K, V
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>


// ============================================================
// Portable BF16 ↔ FP32 conversion helpers
// ============================================================
__device__ __forceinline__ float bf16_to_fp32(__nv_bfloat16 val) {
    unsigned int bits = static_cast<unsigned int>(
        *reinterpret_cast<const unsigned short*>(&val)) << 16;
    return *reinterpret_cast<float*>(&bits);
}

__device__ __forceinline__ __nv_bfloat16 fp32_to_bf16(float val) {
    unsigned int bits = *reinterpret_cast<unsigned int*>(&val);
    unsigned int rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    unsigned short result = static_cast<unsigned short>(bits >> 16);
    return *reinterpret_cast<__nv_bfloat16*>(&result);
}


// ============================================================
// Warp-level sum reduction (lane 0 gets the result)
// ============================================================
__device__ __forceinline__ float warp_sum(float val) {
    const unsigned int mask = 0xFFFFFFFF;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}


// ============================================================
// Block-level sum reduction for LayerNorm
// ============================================================
__device__ float block_sum(float val, float* smem) {
    const int lane  = threadIdx.x % 32;
    const int warp  = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;

    val = warp_sum(val);

    if (lane == 0) smem[warp] = val;
    __syncthreads();

    float warp_val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) warp_val = warp_sum(warp_val);

    if (threadIdx.x == 0) smem[0] = warp_val;
    __syncthreads();

    return smem[0];
}


// ============================================================
// Fused LayerNorm + QKV kernel — FP32 (Phase 5: warp-parallel GEMV)
// ============================================================
//
// Shared memory layout:
//   smem[0 .. 31]:      reduction scratch
//   smem[32 .. 32+D-1]: x_norm for GEMV reads
//
__global__ void fused_ln_qkv_fp32_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ W_qkv,
    const float* __restrict__ b_qkv,
          float* __restrict__ OUT,
    int N,
    int D,
    float eps
) {
    extern __shared__ float smem_raw[];

    float* reduce_smem = smem_raw;
    float* xnorm_smem  = smem_raw + 32;

    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int NUM_WARPS = blockDim.x / 32;   // = 8 for D=256

    if (row >= N || tid >= D) return;

    // ================================================================
    // Phase 1: LayerNorm
    // ================================================================

    float x_val = X[row * D + tid];

    // Mean
    float mean = block_sum(x_val, reduce_smem) / (float)D;

    // Variance
    float diff = x_val - mean;
    float var = block_sum(diff * diff, reduce_smem) / (float)D;

    // Normalize + affine → stays on-chip in shared memory
    float x_hat = diff / sqrtf(var + eps);
    float x_norm = gamma[tid] * x_hat + beta[tid];

    xnorm_smem[tid] = x_norm;
    __syncthreads();

    // ================================================================
    // Phase 2: Warp-parallel GEMV
    //
    // Each warp computes one output column per outer iteration.
    // 8 warps → 8 columns per iteration → 768/8 = 96 iterations.
    //
    // Within each warp, 32 threads cooperatively compute one dot product:
    //   out[col] = sum_k(xnorm[k] * W[col][k]) + bias[col]
    //
    // Thread t handles k = t, t+32, t+64, ..., t+224 (8 elements)
    // Warp reads are COALESCED: thread t reads W[col*D + t], which is
    // contiguous with thread t+1 reading W[col*D + t+1].
    // ================================================================

    const int D3 = 3 * D;

    for (int base_col = 0; base_col < D3; base_col += NUM_WARPS) {
        int out_col = base_col + warp_id;

        if (out_col < D3) {
            const float* w_row = W_qkv + out_col * D;

            // Cooperative dot product: each thread accumulates D/32 partial products
            float partial = 0.0f;
            for (int k = lane_id; k < D; k += 32) {
                partial += xnorm_smem[k] * w_row[k];
            }

            // Warp-level reduction (only lane 0 gets the full sum)
            partial = warp_sum(partial);

            // Lane 0 writes the output
            if (lane_id == 0) {
                OUT[row * D3 + out_col] = partial + b_qkv[out_col];
            }
        }
    }
}


// ============================================================
// Fused LayerNorm + QKV kernel — BF16 (Phase 5: warp-parallel GEMV)
// ============================================================
__global__ void fused_ln_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ W_qkv,
    const __nv_bfloat16* __restrict__ b_qkv,
          __nv_bfloat16* __restrict__ OUT,
    int N,
    int D,
    float eps
) {
    extern __shared__ float smem_raw[];

    float* reduce_smem = smem_raw;
    float* xnorm_smem  = smem_raw + 32;

    const int row     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int NUM_WARPS = blockDim.x / 32;

    if (row >= N || tid >= D) return;

    // ================================================================
    // Phase 1: LayerNorm (FP32 accumulation)
    // ================================================================

    float x_val = bf16_to_fp32(X[row * D + tid]);

    float mean = block_sum(x_val, reduce_smem) / (float)D;

    float diff = x_val - mean;
    float var = block_sum(diff * diff, reduce_smem) / (float)D;

    float x_hat = diff / sqrtf(var + eps);
    float g = bf16_to_fp32(gamma[tid]);
    float b = bf16_to_fp32(beta[tid]);
    float x_norm = g * x_hat + b;

    // Store as FP32 in shared memory for dot product precision
    xnorm_smem[tid] = x_norm;
    __syncthreads();

    // ================================================================
    // Phase 2: Warp-parallel GEMV (FP32 accumulation, BF16 output)
    // ================================================================

    const int D3 = 3 * D;

    for (int base_col = 0; base_col < D3; base_col += NUM_WARPS) {
        int out_col = base_col + warp_id;

        if (out_col < D3) {
            const __nv_bfloat16* w_row = W_qkv + out_col * D;

            // Cooperative dot product with FP32 accumulation
            float partial = 0.0f;
            for (int k = lane_id; k < D; k += 32) {
                partial += xnorm_smem[k] * bf16_to_fp32(w_row[k]);
            }

            // Warp reduction
            partial = warp_sum(partial);

            // Lane 0 writes as BF16
            if (lane_id == 0) {
                float result = partial + bf16_to_fp32(b_qkv[out_col]);
                OUT[row * D3 + out_col] = fp32_to_bf16(result);
            }
        }
    }
}


// ============================================================
// C-linkage launcher functions
// ============================================================

extern "C" {

void launch_fused_ln_qkv_fp32(
    const float* X, const float* gamma, const float* beta,
    const float* W_qkv, const float* b_qkv, float* OUT,
    int N, int D, float eps, cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    int smem_size = (32 + D) * sizeof(float);   // 32 for reduction + D for x_norm
    fused_ln_qkv_fp32_kernel<<<grid, block, smem_size, stream>>>(
        X, gamma, beta, W_qkv, b_qkv, OUT, N, D, eps
    );
}

void launch_fused_ln_qkv_bf16(
    const __nv_bfloat16* X, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const __nv_bfloat16* W_qkv,
    const __nv_bfloat16* b_qkv, __nv_bfloat16* OUT,
    int N, int D, float eps, cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(D);
    int smem_size = (32 + D) * sizeof(float);   // smem is FP32 even for BF16 path
    fused_ln_qkv_bf16_kernel<<<grid, block, smem_size, stream>>>(
        X, gamma, beta, W_qkv, b_qkv, OUT, N, D, eps
    );
}

}  // extern "C"
