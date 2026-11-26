#!/usr/bin/env bash
set -euo pipefail

FLAVOR="${1:-cuda}"
BUILD_TYPE="${2:-Release}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_ROOT}/.." && pwd)"

: "${PARAS_HOME:?Please set PARAS_HOME}"
: "${PTSYCL_GCC_TOOLCHAIN:?Please set PTSYCL_GCC_TOOLCHAIN}"
: "${PTSYCL_LLVM_DIR:?Please set PTSYCL_LLVM_DIR}"

export PTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME:-/usr/local/cuda}"
export PTSYCL_ROCM_HOME="${PTSYCL_ROCM_HOME:-/opt/rocm}"
export PATH="${PTSYCL_LLVM_DIR}/bin:${PARAS_HOME}/bin:${PTSYCL_CUDA_HOME}/bin:${PTSYCL_ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${PTSYCL_GCC_TOOLCHAIN}/lib64:${PTSYCL_LLVM_DIR}/lib:${PARAS_HOME}/lib:${PTSYCL_CUDA_HOME}/lib64:${PTSYCL_ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

case "${FLAVOR}" in
    cpu)  DEVICE="cpu" ;;
    cuda) DEVICE="sycl:${PTSYCL_CUDA_ARCH:-cuda:sm_70}" ;;
    hip)
        if [[ -n "${PTSYCL_HIP_ARCH:-}" ]]; then
            DEVICE="sycl:${PTSYCL_HIP_ARCH}"
        elif command -v rocm_agent_enumerator >/dev/null 2>&1; then
            mapfile -t _gfx_agents < <(rocm_agent_enumerator | grep -v '^gfx000$' | sort -u)
            if [[ ${#_gfx_agents[@]} -eq 0 ]]; then
                echo "error: rocm_agent_enumerator found no AMD GPU agents; set PTSYCL_HIP_ARCH=hip:gfxXXX" >&2
                exit 1
            fi
            DEVICE="sycl:hip:${_gfx_agents[0]}"
        else
            echo "error: set PTSYCL_HIP_ARCH=hip:gfxXXX (e.g. hip:gfx942)" >&2
            exit 1
        fi
        ;;
    *) echo "usage: build_pjrt.sh [cpu|cuda|hip] [Release|Debug]" >&2; exit 1 ;;
esac

BUILD_DIR="${PLUGIN_ROOT}/build/${FLAVOR}"
mkdir -p "${BUILD_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"

"${CMAKE_BIN}" -S "${PLUGIN_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER="${PARAS_HOME}/bin/parascc" \
    -DPTSYCL_DEVICE="${DEVICE}" \
    -DPTSYCL_GCC_TOOLCHAIN="${PTSYCL_GCC_TOOLCHAIN}" \
    -DPARAS_HOME="${PARAS_HOME}" \
    -DPTSYCL_CLAP_ROOT="${PTSYCL_CLAP_ROOT:-${CLAP_HOME:-}}" \
    -DPTSYCL_CUDA_HOME="${PTSYCL_CUDA_HOME:-}" \
    -DPTSYCL_ROCM_HOME="${PTSYCL_ROCM_HOME:-}" \
    -DPTSYCL_PARAS_RUNTIME_DIR="${PTSYCL_PARAS_RUNTIME_DIR:-}"

# Serial build (-j1) required for parascc temp-file safety.
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -j1

cp "${BUILD_DIR}/lib/libpjrt_paras.so" "${PLUGIN_ROOT}/python/jax_paras_plugin/libpjrt_paras.so"
echo "[build_pjrt.sh] installed ${FLAVOR} -> python/jax_paras_plugin/libpjrt_paras.so"

echo "[build_pjrt.sh] checking linkage (must have NO libtorch/libc10):"
if ldd "${PLUGIN_ROOT}/python/jax_paras_plugin/libpjrt_paras.so" | grep -E "libtorch|libc10"; then
    echo "ERROR: libtorch or libc10 found in libpjrt_paras.so!" >&2
    exit 1
fi
nm -D "${PLUGIN_ROOT}/python/jax_paras_plugin/libpjrt_paras.so" | grep GetPjrtApi
