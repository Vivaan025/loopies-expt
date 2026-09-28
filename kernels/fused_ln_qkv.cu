/*
 * Phase 4 — Fused LayerNorm + QKV Projection kernel.
 *
 * This is a CORRECTNESS PROTOTYPE. It is intentionally simple:
 *   - One block per row (per token)
 *   - 256 threads per block (one per feature)
 *   - After LayerNorm, x_norm stays in shared memory
 *   - Each thread computes 3 output values (Q_i, K_i, V_i) via dot product
 *
 * The key benefit: x_norm is NEVER written to global memory (DRAM).
 * It goes from registers → shared memory → dot product → output.
 *
 * Performance notes:
 *   - The dot product loop (D iterations per output) has no weight reuse
 *     between threads and does not use Tensor Cores.
 *   - This is NOT expected to beat cuBLAS/CUTLASS for the GEMM portion.
 *   - The paper contribution is the eliminated DRAM round-trip for x_norm.
 *   - A tiled version (Phase 5) would address GEMM efficiency.
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
// Portable BF16 ↔ FP32 conversion helpers (same as layernorm_kernel.cu)
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
// Warp-level sum reduction
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
// Block-level sum reduction using shared memory
// smem must have at least (blockDim.x / 32) floats available
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
// Fused LayerNorm + QKV kernel — FP32
// ============================================================
//
// Shared memory layout:
//   smem[0 .. 31]:      reduction scratch (up to 32 warps)
//   smem[32 .. 32+D-1]: x_norm storage for dot product
//
__global__ void fused_ln_qkv_fp32_kernel(
    const float* __restrict__ X,        // (N, D)
    const float* __restrict__ gamma,    // (D,)
    const float* __restrict__ beta,     // (D,)
    const float* __restrict__ W_qkv,    // (3D, D) — row-major
    const float* __restrict__ b_qkv,    // (3D,)
          float* __restrict__ OUT,      // (N, 3D)
    int N,
    int D,
    float eps
) {
    extern __shared__ float smem_raw[];

    // Separate regions to avoid reuse hazards
    float* reduce_smem = smem_raw;           // [0..31]  for reductions
    float* xnorm_smem  = smem_raw + 32;     // [32..32+D-1] for x_norm

    const int row = blockIdx.x;
    const int i   = threadIdx.x;

    if (row >= N || i >= D) return;

    // ---- Step 1: Load input ----
    float x_val = X[row * D + i];

    // ---- Step 2: Compute mean ----
    float mean = block_sum(x_val, reduce_smem) / (float)D;

    // ---- Step 3: Compute variance ----
    float diff = x_val - mean;
    float var = block_sum(diff * diff, reduce_smem) / (float)D;

    // ---- Step 4: Normalize + affine ----
    float x_hat = diff / sqrtf(var + eps);
    float x_norm = gamma[i] * x_hat + beta[i];

    // ---- Step 5: Store x_norm in shared memory ----
    // This is the KEY FUSION POINT: x_norm stays on-chip.
    // In the unfused path, x_norm would be written to DRAM here
    // and read back by the QKV projection.
    xnorm_smem[i] = x_norm;
    __syncthreads();

    // ---- Step 6: QKV dot product ----
    // Thread i computes output columns: i, i+D, i+2D
    // Each requires a dot product of xnorm_smem[0..D-1] with one row of W_qkv
    const int D3 = 3 * D;

    for (int j = 0; j < 3; j++) {
        int out_col = j * D + i;
        const float* w_row = W_qkv + out_col * D;   // row out_col of W_qkv

        float dot = 0.0f;
        for (int k = 0; k < D; k++) {
            dot += xnorm_smem[k] * w_row[k];
        }
        OUT[row * D3 + out_col] = dot + b_qkv[out_col];
    }
}


// ============================================================
// Fused LayerNorm + QKV kernel — BF16
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

    const int row = blockIdx.x;
    const int i   = threadIdx.x;

    if (row >= N || i >= D) return;

    // Load as FP32
    float x_val = bf16_to_fp32(X[row * D + i]);

    // Mean
    float mean = block_sum(x_val, reduce_smem) / (float)D;

    // Variance
    float diff = x_val - mean;
    float var = block_sum(diff * diff, reduce_smem) / (float)D;

    // Normalize + affine
    float x_hat = diff / sqrtf(var + eps);
    float g = bf16_to_fp32(gamma[i]);
    float b = bf16_to_fp32(beta[i]);
    float x_norm = g * x_hat + b;

    // Store normalized value in shared memory (FP32 for dot product precision)
    xnorm_smem[i] = x_norm;
    __syncthreads();

    // QKV dot product — accumulate in FP32, store as BF16
    const int D3 = 3 * D;

    for (int j = 0; j < 3; j++) {
        int out_col = j * D + i;
        const __nv_bfloat16* w_row = W_qkv + out_col * D;

        float dot = 0.0f;
        for (int k = 0; k < D; k++) {
            dot += xnorm_smem[k] * bf16_to_fp32(w_row[k]);
        }
        dot += bf16_to_fp32(b_qkv[out_col]);
        OUT[row * D3 + out_col] = fp32_to_bf16(dot);
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
    int smem_size = (32 + D) * sizeof(float);  // 32 for reduction + D for x_norm
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
    int smem_size = (32 + D) * sizeof(float);  // 32 for reduction + D for x_norm   // smem is FP32 even for BF16 path
    fused_ln_qkv_bf16_kernel<<<grid, block, smem_size, stream>>>(
        X, gamma, beta, W_qkv, b_qkv, OUT, N, D, eps
    );
}

}  // extern "C"
