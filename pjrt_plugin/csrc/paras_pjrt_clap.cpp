#include "paras_pjrt_sycl.hpp"

#include <clap/blas_factory.hpp>

#if defined(PARAS_PJRT_CUDA)
#include <cuda_runtime_api.h>
#elif defined(PARAS_PJRT_HIP)
#include <hip/hip_runtime_api.h>
#endif

#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace paras_pjrt {
namespace {

std::mutex& clap_mutex() {
    static std::mutex m;
    return m;
}

std::unique_ptr<clap::IBlasBackend>& clap_backend() {
    static std::unique_ptr<clap::IBlasBackend> instance;
    return instance;
}

void ensure_clap_backend_env(bool is_gpu) {
    // Only set CLAP_BACKEND if the user hasn't explicitly overridden it.
    // Note: CLAP recognizes "AMD" (not "ROCM") for HIP/ROCm!
    if (std::getenv("CLAP_BACKEND") != nullptr) {
        return;
    }
    if (!is_gpu) {
        setenv("CLAP_BACKEND", "CPU", 1);
        return;
    }
#if defined(PARAS_PJRT_CUDA)
    setenv("CLAP_BACKEND", "CUDA", 1);
#elif defined(PARAS_PJRT_HIP)
    setenv("CLAP_BACKEND", "AMD", 1);
#else
    setenv("CLAP_BACKEND", "CPU", 1);
#endif
}

clap::IBlasBackend* get_or_create_backend(bool is_gpu, std::string* err_msg) {
    std::lock_guard<std::mutex> lock(clap_mutex());
    auto& backend = clap_backend();
    if (!backend) {
        ensure_clap_backend_env(is_gpu);
        try {
            backend = clap::BlasFactory::create();
        } catch (const std::exception& e) {
            if (err_msg) {
                *err_msg = std::string("clap::BlasFactory::create() threw: ") + e.what();
            }
            return nullptr;
        }
        if (!backend && err_msg) {
            *err_msg = "clap::BlasFactory::create() returned nullptr";
        }
    }
    return backend.get();
}

} // namespace

bool clap_init_backend(bool is_gpu, std::string* err_msg) {
    return get_or_create_backend(is_gpu, err_msg) != nullptr;
}

bool clap_gemm_f32_raw(const void* a, const void* b, void* c,
                       int m, int n, int k,
                       bool trans_a, bool trans_b,
                       float alpha, float beta,
                       std::string* err_msg) {
    if (m == 0 || n == 0) {
        return true;
    }

    clap::IBlasBackend* backend = get_or_create_backend(true, err_msg);
    if (!backend) {
        return false;
    }

    // RowMajor(C[M,N] = op_a(A) @ op_b(B)) == ColMajor(C^T[N,M] = op_b(B)^T @ op_a(A)^T).
    const clap::Transpose first_trans =
        trans_b ? clap::Transpose::Trans : clap::Transpose::NoTrans;
    const clap::Transpose second_trans =
        trans_a ? clap::Transpose::Trans : clap::Transpose::NoTrans;

    const int lda = trans_b ? k : n; // leading dim of B viewed as ColMajor
    const int ldb = trans_a ? m : k; // leading dim of A viewed as ColMajor
    const int ldc = n;               // leading dim of C viewed as ColMajor (N x M)

    try {
        backend->gemm(
            clap::Layout::ColMajor,
            first_trans,
            second_trans,
            n, m, k,
            alpha,
            static_cast<const float*>(b), lda,
            static_cast<const float*>(a), ldb,
            beta,
            static_cast<float*>(c), ldc);
    } catch (const std::exception& e) {
        if (err_msg) {
            *err_msg = std::string("CLAP gemm failed: ") + e.what();
        }
        return false;
    }

#if defined(PARAS_PJRT_CUDA)
    cudaError_t cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        if (err_msg) {
            *err_msg = std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(cerr);
        }
        return false;
    }
#elif defined(PARAS_PJRT_HIP)
    hipError_t herr = hipDeviceSynchronize();
    if (herr != hipSuccess) {
        if (err_msg) {
            *err_msg = std::string("hipDeviceSynchronize failed: ") + hipGetErrorString(herr);
        }
        return false;
    }
#endif

    return true;
}

} // namespace paras_pjrt
