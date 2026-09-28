/*
 * PyTorch C++ extension binding for custom CUDA kernels.
 *
 * Exposes:
 *   - custom_layernorm(X, gamma, beta, eps) → Y
 *   - fused_ln_qkv(X, gamma, beta, W_qkv, b_qkv, eps) → OUT
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


// ============================================================
// Forward declarations — LayerNorm (layernorm_kernel.cu)
// ============================================================
extern "C" {
    void launch_layernorm_fp32(
        const float* X, const float* gamma, const float* beta, float* Y,
        int N, int D, float eps, cudaStream_t stream
    );
    void launch_layernorm_bf16(
        const __nv_bfloat16* X, const __nv_bfloat16* gamma,
        const __nv_bfloat16* beta, __nv_bfloat16* Y,
        int N, int D, float eps, cudaStream_t stream
    );
}

// ============================================================
// Forward declarations — Fused LN+QKV (fused_ln_qkv.cu)
// ============================================================
extern "C" {
    void launch_fused_ln_qkv_fp32(
        const float* X, const float* gamma, const float* beta,
        const float* W_qkv, const float* b_qkv, float* OUT,
        int N, int D, float eps, cudaStream_t stream
    );
    void launch_fused_ln_qkv_bf16(
        const __nv_bfloat16* X, const __nv_bfloat16* gamma,
        const __nv_bfloat16* beta, const __nv_bfloat16* W_qkv,
        const __nv_bfloat16* b_qkv, __nv_bfloat16* OUT,
        int N, int D, float eps, cudaStream_t stream
    );
}


// ============================================================
// custom_layernorm: standalone LayerNorm
// ============================================================
torch::Tensor custom_layernorm(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    TORCH_CHECK(X.is_cuda(),           "X must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),       "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(),        "beta must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),  "beta must be contiguous");

    auto orig_shape = X.sizes().vec();
    int D = X.size(-1);
    int N = X.numel() / D;

    TORCH_CHECK(gamma.numel() == D, "gamma must have D elements");
    TORCH_CHECK(beta.numel()  == D, "beta must have D elements");
    TORCH_CHECK(D <= 1024, "D must be <= 1024 (one thread per feature)");

    auto X_flat = X.reshape({N, D});
    auto Y = torch::empty_like(X_flat);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (X.dtype() == torch::kFloat32) {
        launch_layernorm_fp32(
            X_flat.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            Y.data_ptr<float>(),
            N, D, (float)eps, stream
        );
    } else if (X.dtype() == torch::kBFloat16) {
        launch_layernorm_bf16(
            reinterpret_cast<const __nv_bfloat16*>(X_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            N, D, (float)eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", X.dtype());
    }

    return Y.reshape(orig_shape);
}


// ============================================================
// fused_ln_qkv: LayerNorm + QKV projection in one kernel
// ============================================================
torch::Tensor fused_ln_qkv(
    torch::Tensor X,        // (*, D) — input activations
    torch::Tensor gamma,    // (D,)   — LayerNorm weight
    torch::Tensor beta,     // (D,)   — LayerNorm bias
    torch::Tensor W_qkv,    // (3D, D) — packed QKV weight
    torch::Tensor b_qkv,    // (3D,)   — packed QKV bias
    double eps
) {
    // --- Validation ---
    TORCH_CHECK(X.is_cuda(),       "X must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),   "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(),    "beta must be a CUDA tensor");
    TORCH_CHECK(W_qkv.is_cuda(),   "W_qkv must be a CUDA tensor");
    TORCH_CHECK(b_qkv.is_cuda(),   "b_qkv must be a CUDA tensor");

    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),  "beta must be contiguous");
    TORCH_CHECK(W_qkv.is_contiguous(), "W_qkv must be contiguous");
    TORCH_CHECK(b_qkv.is_contiguous(), "b_qkv must be contiguous");

    int D = X.size(-1);
    int N = X.numel() / D;
    int D3 = 3 * D;

    TORCH_CHECK(gamma.numel() == D,      "gamma must have D elements");
    TORCH_CHECK(beta.numel()  == D,      "beta must have D elements");
    TORCH_CHECK(W_qkv.size(0) == D3,    "W_qkv must have shape (3D, D)");
    TORCH_CHECK(W_qkv.size(1) == D,     "W_qkv must have shape (3D, D)");
    TORCH_CHECK(b_qkv.numel() == D3,    "b_qkv must have 3D elements");
    TORCH_CHECK(D <= 1024, "D must be <= 1024");

    // Flatten input to (N, D)
    auto X_flat = X.reshape({N, D});

    // Output: (N, 3D)
    auto OUT = torch::empty({N, D3}, X.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (X.dtype() == torch::kFloat32) {
        launch_fused_ln_qkv_fp32(
            X_flat.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            W_qkv.data_ptr<float>(),
            b_qkv.data_ptr<float>(),
            OUT.data_ptr<float>(),
            N, D, (float)eps, stream
        );
    } else if (X.dtype() == torch::kBFloat16) {
        launch_fused_ln_qkv_bf16(
            reinterpret_cast<const __nv_bfloat16*>(X_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(W_qkv.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_qkv.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr<at::BFloat16>()),
            N, D, (float)eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", X.dtype());
    }

    // Reshape to (*, 3D) preserving batch dims
    auto out_shape = X.sizes().vec();
    out_shape.back() = D3;
    return OUT.reshape(out_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &custom_layernorm,
          "Custom LayerNorm (CUDA)",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("eps") = 1e-5);

    m.def("fused_ln_qkv", &fused_ln_qkv,
          "Fused LayerNorm + QKV projection (CUDA)",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("W_qkv"),
          py::arg("b_qkv"),
          py::arg("eps") = 1e-5);
}
