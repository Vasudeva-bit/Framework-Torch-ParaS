# `pjrt_paras` v0.1 — JAX PJRT C-API Plugin Prototype

Prototype JAX PJRT C-API plugin (`libpjrt_paras.so` + `jax_paras_plugin`) executing 2D `float32` MatMul (`jax.jit(lambda a, b: a @ b)(a, b)`) on NVIDIA CUDA and AMD HIP/ROCm GPUs via `ParaS-Compiler` (`parascc`) and `CLAP-Library`.

## Design Rules
- **Zero `libtorch` / `libc10` / `pybind11` linkage** in `libpjrt_paras.so`.
- **Direct PJRT C API** (`GetPjrtApi()`).
- **Serial builds only (`-j1`)** when invoking `parascc`.
- **Single-runtime rule:** never `import torch` in the same Python process as JAX.

## Quick Start (on GPU build/test node)

1. **Fetch `pjrt_c_api.h` matching your installed `jaxlib`:**
   ```bash
   ./pjrt_plugin/scripts/fetch_pjrt_header.sh main
   ```

2. **Build `libpjrt_paras.so` (`cuda`, `hip`, or `cpu`):**
   ```bash
   export PARAS_HOME=/path/to/paras
   export PTSYCL_GCC_TOOLCHAIN=/path/to/gcc
   export PTSYCL_LLVM_DIR=/path/to/llvm
   export CLAP_HOME=/path/to/clap
   # For CUDA: export PTSYCL_CUDA_HOME=/usr/local/cuda PTSYCL_CUDA_ARCH=cuda:sm_80
   # For HIP:  export PTSYCL_ROCM_HOME=/opt/rocm PTSYCL_HIP_ARCH=hip:gfx942

   bash pjrt_plugin/scripts/build_pjrt.sh cuda Release
   ```

3. **Install `jax_paras_plugin` into the JAX Python environment:**
   ```bash
   pip install --no-build-isolation --no-deps -e ./pjrt_plugin
   ```

4. **Run the MatMul verification:**
   ```bash
   XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=paras python3 pjrt_plugin/tests/test_matmul.py
   ```
