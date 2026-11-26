#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace paras_pjrt {

struct ParasDeviceInfo {
    int         id               = 0;
    char        name[256]        = {};
    int         compute_units    = 0;
    std::size_t global_mem_bytes = 0;
    bool        is_gpu           = false;
};

// Initializes SYCL runtime (or CPU fallback) and CLAP backend.
// Returns 0 on success, non-zero on error (with message in err_msg).
int init_runtime(std::string* err_msg);

int device_count();
bool get_device_info(int device_id, ParasDeviceInfo* out);

// USM / host memory allocation and transfers
void* alloc_device(int device_id, std::size_t nbytes, std::string* err_msg);
void  free_device(int device_id, void* ptr);

bool copy_h2d(int device_id, void* dst_dev, const void* src_host,
              std::size_t nbytes, bool blocking, std::string* err_msg);
bool copy_d2h(int device_id, void* dst_host, const void* src_dev,
              std::size_t nbytes, bool blocking, std::string* err_msg);
bool copy_d2d(int device_id, void* dst_dev, const void* src_dev,
              std::size_t nbytes, std::string* err_msg);
bool memset_device(int device_id, void* dst_dev, int value,
                   std::size_t nbytes, std::string* err_msg);
bool synchronize_device(int device_id, std::string* err_msg);

// Synchronizes SYCL queue before & after calling CLAP BLAS GEMM:
//   C[M, N] = alpha * op_a(A) @ op_b(B) + beta * C[M, N]  (row-major f32)
bool gemm_f32(int device_id,
              const void* a, const void* b, void* c,
              int m, int n, int k,
              bool trans_a, bool trans_b,
              float alpha, float beta,
              std::string* err_msg);

// Implemented in paras_pjrt_clap.cpp (compiled with host g++).
bool clap_init_backend(bool is_gpu, std::string* err_msg);
bool clap_gemm_f32_raw(const void* a, const void* b, void* c,
                       int m, int n, int k,
                       bool trans_a, bool trans_b,
                       float alpha, float beta,
                       std::string* err_msg);

} // namespace paras_pjrt
