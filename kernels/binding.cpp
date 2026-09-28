/*
 * PyTorch C++ extension binding for custom CUDA kernels.
 *
 * Exposes:
 *   - custom_layernorm(X, gamma, beta, eps) → Y
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


// Forward declarations of launcher functions from .cu files
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


torch::Tensor custom_layernorm(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    // --- Input validation ---
    TORCH_CHECK(X.is_cuda(),      "X must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(),  "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(),   "beta must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),     "X must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),  "beta must be contiguous");

    // Support (N, D) or (B, S, D) — flatten leading dims
    auto orig_shape = X.sizes().vec();
    int D = X.size(-1);
    int N = X.numel() / D;

    TORCH_CHECK(gamma.numel() == D, "gamma must have D elements");
    TORCH_CHECK(beta.numel()  == D, "beta must have D elements");
    TORCH_CHECK(D <= 1024, "D must be <= 1024 (one thread per feature)");

    // Flatten to (N, D)
    auto X_flat = X.reshape({N, D});
    auto Y = torch::empty_like(X_flat);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch by dtype
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
        TORCH_CHECK(false, "Unsupported dtype: ", X.dtype(), ". Use float32 or bfloat16.");
    }

    // Reshape back to original shape
    return Y.reshape(orig_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &custom_layernorm,
          "Custom LayerNorm (CUDA)",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("eps") = 1e-5);
}
