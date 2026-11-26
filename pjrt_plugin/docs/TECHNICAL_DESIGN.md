# `pjrt_paras` v0.1 Technical Documentation

This document describes the architecture, build system, runtime components, and design constraints of the `pjrt_paras` v0.1 JAX PJRT C-API prototype implemented under `pjrt_plugin/`.

---

## 1. Design Goals & Hard Constraints

- **Standalone Linkage (Zero PyTorch Dependencies):** `libpjrt_paras.so` links **no** `libtorch`, `libc10`, or `pybind11` libraries. Keeping it isolated under `pjrt_plugin/` allows fast prototyping alongside `Framework-Torch-ParaS` while enabling a clean split into a standalone repository at v1.0.
- **Option A (Direct PJRT C API):** Implements `pjrt_c_api.h` directly in C++ and exports `GetPjrtApi()`, without building or linking `libxla` or `pjrt_c_api_wrapper_impl.h`.
- **Single-Runtime Rule:** `libpjrt_paras.so` must be built against the same CUDA or ROCm/HIP toolkit used by `ParaS-Compiler` (`parascc`) and `CLAP-Library`, and `torch` must never be imported in the same Python process as `jax`.
- **Serial Compilation (`-j1`):** `parascc` uses second-resolution temporary filenames in `/tmp`; all `parascc` builds must run serially (`-- -j1`).

---

## 2. Directory Layout

```text
pjrt_plugin/
├── CMakeLists.txt                       # Standalone CMake target for libpjrt_paras.so
├── README.md                            # Quick-start build and run instructions
├── pyproject.toml                       # Registers jax_plugins entry point "paras"
├── docs/
│   └── TECHNICAL_DESIGN.md              # This technical documentation
├── scripts/
│   ├── fetch_pjrt_header.sh             # Downloads xla/pjrt/c/pjrt_c_api.h into third_party/
│   └── build_pjrt.sh                    # Configures, builds (-j1), and verifies libpjrt_paras.so
├── third_party/
│   └── pjrt_c_api.h                     # (Fetched) OpenXLA PJRT C-API header matching jaxlib
├── csrc/
│   ├── paras_pjrt_sycl.hpp              # Pure C++ runtime/allocator/GEMM interface (no SYCL/CLAP headers)
│   ├── paras_pjrt_sycl.cpp              # SYCL device/queue/USM implementation (compiled by parascc)
│   ├── paras_pjrt_clap.cpp              # CLAP-BLAS GEMM wrapper (compiled by host g++)
│   ├── pjrt_types.hpp                   # Concrete structs backing PJRT_* opaque handles
│   ├── pjrt_event.cpp                   # PJRT_Event_* lifecycle and OnReady callbacks
│   ├── pjrt_buffer.cpp                  # Host<->Device buffer transfers and metadata
│   ├── pjrt_compile.cpp                 # StableHLO MLIR parsing -> MatMulPlan
│   ├── pjrt_executable.cpp              # PJRT_Executable_* / PJRT_LoadedExecutable_* & GEMM dispatch
│   └── pjrt_api.cpp                     # PJRT_Api table population and GetPjrtApi() export
├── python/
│   └── jax_paras_plugin/
│       └── __init__.py                  # initialize() -> xb.register_plugin("paras", ...)
└── tests/
    └── test_matmul.py                   # End-to-end JAX jit MatMul verification
```

---

## 3. Build System (`CMakeLists.txt` & `scripts/build_pjrt.sh`)

### Two-Compiler Split (`parascc` vs. Host `g++`)
1. **`parascc` (`CMAKE_CXX_COMPILER=${PARAS_HOME}/bin/parascc`):**
   Compiles `csrc/paras_pjrt_sycl.cpp` and all `csrc/pjrt_*.cpp` translation units individually into object files. Only `paras_pjrt_sycl.cpp` includes `<sycl/sycl.hpp>`, keeping SYCL types out of headers because `parascc` rewrites the main `.cpp` translation unit rather than included headers.
2. **Host `g++` (`${PTSYCL_GCC_TOOLCHAIN}/bin/g++`):**
   Compiles `csrc/paras_pjrt_clap.cpp` via an `add_custom_command` rule into `paras_pjrt_clap.o`, isolating `<clap/blas_factory.hpp>` and vendor runtime headers (`cuda_runtime_api.h` / `hip/hip_runtime_api.h`) from `parascc`.

### Backend Flavors (`PTSYCL_DEVICE`)
| `build_pjrt.sh` flavor | `PTSYCL_DEVICE` | Compile Definitions | GPU Runtime Linked |
|---|---|---|---|
| `cpu` | `cpu` | `PARAS_PJRT_BACKEND_CPU` | None |
| `cuda` | `sycl:cuda:sm_XX` | `PARAS_PJRT_BACKEND_SYCL`, `PARAS_PJRT_CUDA` | `libcudart` (from `PTSYCL_CUDA_HOME`) |
| `hip` | `sycl:hip:gfxXXX` | `PARAS_PJRT_BACKEND_SYCL`, `PARAS_PJRT_HIP` | `libamdhip64` (if found under `PTSYCL_ROCM_HOME`) |

### Verification Checks in `build_pjrt.sh`
After building with `-- -j1` and copying `libpjrt_paras.so` to `python/jax_paras_plugin/libpjrt_paras.so`, `build_pjrt.sh` runs:
- `ldd libpjrt_paras.so | grep -E "libtorch|libc10"` (fails if any PyTorch library is linked).
- `nm -D libpjrt_paras.so | grep GetPjrtApi` (verifies the PJRT entry symbol is exported).

---

## 4. Runtime Components

### 4.1 SYCL Queue & USM Memory (`paras_pjrt_sycl.hpp` / `paras_pjrt_sycl.cpp`)
- `paras_pjrt::init_runtime()` enumerates `sycl::device::get_devices(sycl::info::device_type::gpu)` under `PARAS_PJRT_BACKEND_SYCL`, creates an in-order `sycl::queue` (`sycl::property::queue::in_order{}`) per GPU, and initializes CLAP via `clap_init_backend()`. If no GPU is present (or in CPU mode), it registers a single `"ParaS CPU Fallback"` device using `posix_memalign` / `std::memcpy`.
- `alloc_device()` / `free_device()` use `sycl::malloc_shared<std::byte>` and `sycl::free` on GPU devices.
- `copy_h2d()`, `copy_d2h()`, `copy_d2d()`, and `memset_device()` wrap `sycl::queue::memcpy` / `memset` and `wait()`.
- `gemm_f32()` synchronizes the SYCL queue (`synchronize_device()`) **before** and **after** calling `clap_gemm_f32_raw()` so pending H2D transfers are visible to CLAP and results are complete before D2H copies.

### 4.2 CLAP-BLAS Wrapper (`paras_pjrt_clap.cpp`)
- **Backend Selection:** `ensure_clap_backend_env()` sets `CLAP_BACKEND` (only if not already set by the user) to `"CUDA"`, `"AMD"` (note: CLAP recognizes `"AMD"`, not `"ROCM"`), or `"CPU"`, avoiding reliance on `/proc/self/cmdline` flags (`-cuda`, `-rocm`, `-cpu`).
- **Row-Major to Column-Major GEMM Mapping:**
  JAX tensors are row-major. To compute row-major $C_{M \times N} = \text{op}_A(A) \cdot \text{op}_B(B)$, `clap_gemm_f32_raw()` invokes `backend->gemm(clap::Layout::ColMajor, ...)` using the transpose identity:
  $$C^T_{N \times M} = (\text{op}_B(B))^T_{N \times K} \cdot (\text{op}_A(A))^T_{K \times M}$$
  - `m_col = n`, `n_col = m`, `k_col = k`
  - First operand: `b` with `first_trans = trans_b ? Trans : NoTrans`, `lda = trans_b ? k : n`
  - Second operand: `a` with `second_trans = trans_a ? Trans : NoTrans`, `ldb = trans_a ? m : k`
  - Output: `c` with `ldc = n`
- **Device Synchronization:** After `backend->gemm(...)`, `clap_gemm_f32_raw()` calls `cudaDeviceSynchronize()` (when `PARAS_PJRT_CUDA` is defined) or `hipDeviceSynchronize()` (when `PARAS_PJRT_HIP` is defined) so asynchronous vendor BLAS kernels finish before `PJRT_Buffer_ToHostBuffer` reads the output buffer.

### 4.3 PJRT C-API Implementation (`pjrt_types.hpp`, `pjrt_*.cpp`)
- **`pjrt_event.cpp`:** Implements `PJRT_Event_Destroy`, `PJRT_Event_IsReady`, `PJRT_Event_Error`, `PJRT_Event_Await`, and `PJRT_Event_OnReady`. `event_on_ready` always invokes `args->callback(err_copy, args->user_arg)` immediately if the event is already ready.
- **`pjrt_buffer.cpp`:**
  - `client_buffer_from_host_buffer`: Allocates USM/host memory, packs non-contiguous strided host inputs if `byte_strides` are non-contiguous, copies H2D, and returns a `PJRT_Buffer*` plus a ready `PJRT_Event*`.
  - `buffer_to_host_buffer`: Handles PJRT's two-phase size query (`if (args->dst == nullptr) { args->dst_size = buf->byte_size; return nullptr; }`) and copies D2H on the second call.
  - Implements `Delete`/`Destroy`, `CopyToDevice`, `GetMemoryLayout` (row-major `minor_to_major`), `OnDeviceSizeInBytes`, `UnsafePointer`, `IsOnCpu` (`false`), and `ReadyEvent`.
- **`pjrt_compile.cpp`:**
  - Accepts `PJRT_Program` code in MLIR text or MLIR bytecode (`ML\xefR`).
  - For MLIR bytecode, writes a temporary file and attempts deserialization via `stablehlo-translate --deserialize` or `mlir-opt --allow-unregistered-dialect` on `PATH`. Setting `PARAS_PJRT_DUMP_MLIR=1` prints the deserialized MLIR to `stderr`.
  - Parses 2D `tensor<...xf32>` shapes, `stablehlo.dot_general` / `stablehlo.dot` contracting dimensions (`contracting_dims = [a] x [b]` or `lhs_contracting_dimensions` / `rhs_contracting_dimensions`), and optional `stablehlo.transpose` on `%arg0` / `%arg1` into a `MatMulPlan` (`m`, `n`, `k`, `trans_a`, `trans_b`, `out_dims`).
- **`pjrt_executable.cpp`:**
  - `loaded_executable_execute` validates single-device execution (`num_devices == 1`), 2 `float32` input buffers, allocates the output `PJRT_Buffer` (`M * N * sizeof(float)`), handles `k == 0` via `memset_device(..., 0)`, otherwise dispatches `paras_pjrt::gemm_f32(...)`, and populates `output_lists[0][0]` and `device_complete_events[0]`.
- **`pjrt_api.cpp`:**
  - Implements `PJRT_Error_*`, `PJRT_Plugin_*`, `PJRT_Client_*`, `PJRT_DeviceDescription_*`, `PJRT_Device_*`, and `PJRT_Memory_*`.
  - Exports `extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi()` with `PJRT_Api_STRUCT_SIZE`, `PJRT_Api_Version_STRUCT_SIZE`, `PJRT_API_MAJOR`, and `PJRT_API_MINOR` from `pjrt_c_api.h`.

---

## 5. Python Plugin & Tests

- **`python/jax_paras_plugin/__init__.py`:** Locates `libpjrt_paras.so` (via `PJRT_PARAS_LIBRARY_PATH` or alongside `__init__.py`) and calls `jax._src.xla_bridge.register_plugin("paras", priority=500, library_path=plugin_path, options=None)`.
- **`tests/test_matmul.py`:** Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `JAX_PLATFORMS=paras`, verifies `jax.devices()` exposes a `paras` device, and compares `jax.jit` MatMul against `np.matmul` (`rtol=1e-4, atol=1e-4`) over 3 iterations for `(64, 64) @ (64, 64)`, `(128, 256) @ (256, 64)`, and `(128, 256) @ (64, 256).T`.

---

## 6. Known v0.1 Scope & Limitations

- Supports single-device 2D `float32` MatMul (`dot_general` / `dot` with optional input transpose) only.
- `pjrt_c_api.h` is not vendored in git; it must be fetched via `scripts/fetch_pjrt_header.sh` to match the target node's `jaxlib` PJRT API version.
- MLIR bytecode (`ML\xefR`) deserialization in `pjrt_compile.cpp` requires `stablehlo-translate` or `mlir-opt` on `PATH`.
