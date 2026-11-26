cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(pjrt_paras LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------------------
# Configuration (standalone — ZERO libtorch / libc10 / pybind11 linkage)
# ------------------------------------------------------------------------------
set(PTSYCL_DEVICE "cpu" CACHE STRING "Target: cpu, or sycl:cuda:sm_XX / sycl:hip:gfxXXX")
set(PTSYCL_GCC_TOOLCHAIN "$ENV{PTSYCL_GCC_TOOLCHAIN}" CACHE PATH "GCC toolchain root for parascc")
set(PARAS_HOME "$ENV{PARAS_HOME}" CACHE PATH "ParaS compiler installation")
set(PTSYCL_LLVM_DIR "$ENV{PTSYCL_LLVM_DIR}" CACHE PATH "LLVM/clang installation used by parascc")
set(PTSYCL_CLAP_ROOT "$ENV{CLAP_HOME}" CACHE PATH "CLAP installation prefix")
set(PTSYCL_CUDA_HOME "$ENV{PTSYCL_CUDA_HOME}" CACHE PATH "CUDA toolkit root")
set(PTSYCL_ROCM_HOME "$ENV{PTSYCL_ROCM_HOME}" CACHE PATH "ROCm toolkit root")
set(PTSYCL_PARAS_RUNTIME_DIR "" CACHE PATH "Optional ParaS runtime library override")

set(PJRT_C_API_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt_c_api.h" CACHE FILEPATH
    "Path to xla/pjrt/c/pjrt_c_api.h matching installed jaxlib")

if(NOT EXISTS "${PJRT_C_API_HEADER}")
    message(FATAL_ERROR
        "pjrt_c_api.h not found at ${PJRT_C_API_HEADER}.\n"
        "Run pjrt_plugin/scripts/fetch_pjrt_header.sh or place openxla/xla's "
        "xla/pjrt/c/pjrt_c_api.h (matching your installed jaxlib) at ${PJRT_C_API_HEADER}.")
endif()

if(NOT PARAS_HOME)
    message(FATAL_ERROR "PARAS_HOME must be set")
endif()
if(NOT PTSYCL_GCC_TOOLCHAIN)
    message(FATAL_ERROR "PTSYCL_GCC_TOOLCHAIN must be set")
endif()

# ------------------------------------------------------------------------------
# Flavor selection
# ------------------------------------------------------------------------------
if(PTSYCL_DEVICE STREQUAL "cpu")
    set(PARAS_PJRT_BACKEND_DEFINE PARAS_PJRT_BACKEND_CPU)
    set(PTSYCL_DEVICE_FLAGS "")
elseif(PTSYCL_DEVICE MATCHES "^sycl:(cuda:sm_[0-9]+|hip:gfx[0-9a-zA-Z]+)$")
    set(PARAS_PJRT_BACKEND_DEFINE PARAS_PJRT_BACKEND_SYCL)
    set(PTSYCL_PARASDEVICE_TARGET "${CMAKE_MATCH_1}")
    set(PTSYCL_DEVICE_FLAGS "-parasdevice;${PTSYCL_PARASDEVICE_TARGET}")
else()
    message(FATAL_ERROR "PTSYCL_DEVICE must be 'cpu', 'sycl:cuda:sm_XX', or "
                        "'sycl:hip:gfxXXX', got '${PTSYCL_DEVICE}'")
endif()

message(STATUS "ParaS home      : ${PARAS_HOME}")
message(STATUS "Device flavor   : ${PTSYCL_DEVICE} (${PARAS_PJRT_BACKEND_DEFINE})")
message(STATUS "PJRT header     : ${PJRT_C_API_HEADER}")

# ------------------------------------------------------------------------------
# GPU runtime discovery (single-runtime rule: match ParaS + CLAP toolkit)
# ------------------------------------------------------------------------------
set(PARAS_PJRT_GPU_RUNTIME_LIB "")
set(PARAS_PJRT_GPU_DEFINE "")

if(PARAS_PJRT_BACKEND_DEFINE STREQUAL "PARAS_PJRT_BACKEND_SYCL")
    if(PTSYCL_PARASDEVICE_TARGET MATCHES "^cuda:")
        if(NOT PTSYCL_CUDA_HOME)
            message(FATAL_ERROR "PTSYCL_CUDA_HOME must be set for sycl:cuda:sm_XX")
        endif()
        find_library(PTSYCL_CUDART_LIBRARY NAMES cudart
            HINTS "${PTSYCL_CUDA_HOME}/lib64" "${PTSYCL_CUDA_HOME}/lib")
        if(NOT PTSYCL_CUDART_LIBRARY)
            message(FATAL_ERROR "libcudart not found under PTSYCL_CUDA_HOME=${PTSYCL_CUDA_HOME}")
        endif()
        set(PARAS_PJRT_GPU_RUNTIME_LIB "${PTSYCL_CUDART_LIBRARY}")
        set(PARAS_PJRT_GPU_DEFINE PARAS_PJRT_CUDA)
        message(STATUS "CUDA runtime    : ${PARAS_PJRT_GPU_RUNTIME_LIB}")
    elseif(PTSYCL_PARASDEVICE_TARGET MATCHES "^hip:")
        if(NOT PTSYCL_ROCM_HOME)
            set(PTSYCL_ROCM_HOME "/opt/rocm")
        endif()
        find_library(PTSYCL_AMDHIP_LIBRARY NAMES amdhip64
            HINTS "${PTSYCL_ROCM_HOME}/lib" "${PTSYCL_ROCM_HOME}/lib64")
        if(PTSYCL_AMDHIP_LIBRARY)
            set(PARAS_PJRT_GPU_RUNTIME_LIB "${PTSYCL_AMDHIP_LIBRARY}")
            message(STATUS "HIP runtime     : ${PARAS_PJRT_GPU_RUNTIME_LIB}")
        else()
            message(WARNING "libamdhip64 not found under PTSYCL_ROCM_HOME=${PTSYCL_ROCM_HOME}; continuing without explicit amdhip64 link")
        endif()
        set(PARAS_PJRT_GPU_DEFINE PARAS_PJRT_HIP)
    endif()
endif()

# ------------------------------------------------------------------------------
# CLAP-BLAS discovery (required for v0.1 MatMul)
# ------------------------------------------------------------------------------
set(_clap_include_hints)
set(_clap_library_hints)
if(PTSYCL_CLAP_ROOT)
    list(APPEND _clap_include_hints "${PTSYCL_CLAP_ROOT}/include")
    list(APPEND _clap_library_hints "${PTSYCL_CLAP_ROOT}/lib64" "${PTSYCL_CLAP_ROOT}/lib")
endif()

find_path(PTSYCL_CLAP_INCLUDE_DIR
    NAMES clap/blas_factory.hpp
    HINTS ${_clap_include_hints})

find_library(PTSYCL_CLAP_LIBRARY
    NAMES clap
    HINTS ${_clap_library_hints})

if(NOT PTSYCL_CLAP_INCLUDE_DIR OR NOT PTSYCL_CLAP_LIBRARY)
    message(FATAL_ERROR
        "CLAP headers/library not found. Set CLAP_HOME or PTSYCL_CLAP_ROOT.")
endif()

get_filename_component(PTSYCL_CLAP_LIBRARY_DIR "${PTSYCL_CLAP_LIBRARY}" DIRECTORY)
message(STATUS "CLAP include    : ${PTSYCL_CLAP_INCLUDE_DIR}")
message(STATUS "CLAP library    : ${PTSYCL_CLAP_LIBRARY}")

# ------------------------------------------------------------------------------
# Compile paras_pjrt_clap.cpp with host g++ (keeps CLAP headers separate from parascc)
# ------------------------------------------------------------------------------
set(PARAS_PJRT_CLAP_OBJ "${CMAKE_CURRENT_BINARY_DIR}/paras_pjrt_clap.o")
set(PARAS_PJRT_HOST_CXX "${PTSYCL_GCC_TOOLCHAIN}/bin/g++")
if(NOT EXISTS "${PARAS_PJRT_HOST_CXX}")
    message(FATAL_ERROR "Host C++ compiler not found: ${PARAS_PJRT_HOST_CXX}")
endif()

set(_clap_extra_flags "")
if(PARAS_PJRT_GPU_DEFINE STREQUAL "PARAS_PJRT_CUDA")
    list(APPEND _clap_extra_flags "-DPARAS_PJRT_CUDA" "-I${PTSYCL_CUDA_HOME}/include")
elseif(PARAS_PJRT_GPU_DEFINE STREQUAL "PARAS_PJRT_HIP")
    list(APPEND _clap_extra_flags "-DPARAS_PJRT_HIP" "-D__HIP_PLATFORM_AMD__" "-I${PTSYCL_ROCM_HOME}/include")
endif()

add_custom_command(
    OUTPUT "${PARAS_PJRT_CLAP_OBJ}"
    COMMAND "${PARAS_PJRT_HOST_CXX}"
        -std=c++17
        -fPIC
        -O3
        "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc"
        "-I${PTSYCL_CLAP_INCLUDE_DIR}"
        ${_clap_extra_flags}
        -c "${CMAKE_CURRENT_SOURCE_DIR}/csrc/paras_pjrt_clap.cpp"
        -o "${PARAS_PJRT_CLAP_OBJ}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/paras_pjrt_clap.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/paras_pjrt_sycl.hpp"
    VERBATIM
)

set_source_files_properties(
    "${PARAS_PJRT_CLAP_OBJ}"
    PROPERTIES
        GENERATED TRUE
        EXTERNAL_OBJECT TRUE
)

# ------------------------------------------------------------------------------
# libpjrt_paras.so target (compiled with parascc, serial -j1)
# ------------------------------------------------------------------------------
get_filename_component(PJRT_C_API_DIR "${PJRT_C_API_HEADER}" DIRECTORY)

add_library(pjrt_paras SHARED
    csrc/paras_pjrt_sycl.cpp
    csrc/pjrt_event.cpp
    csrc/pjrt_buffer.cpp
    csrc/pjrt_compile.cpp
    csrc/pjrt_executable.cpp
    csrc/pjrt_api.cpp
    "${PARAS_PJRT_CLAP_OBJ}"
)

set_target_properties(pjrt_paras PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "pjrt_paras"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_compile_definitions(pjrt_paras PRIVATE
    ${PARAS_PJRT_BACKEND_DEFINE}
    ${PARAS_PJRT_GPU_DEFINE}
)

target_include_directories(pjrt_paras PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
)

target_include_directories(pjrt_paras SYSTEM PRIVATE
    "${PJRT_C_API_DIR}"
    "${PARAS_HOME}/include"
)

target_compile_options(pjrt_paras PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
    ${PTSYCL_DEVICE_FLAGS}
    -Wall
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O0;-g>
)

target_link_directories(pjrt_paras PRIVATE
    "${PTSYCL_GCC_TOOLCHAIN}/lib64"
)

file(GLOB _gcc_lib_dirs "${PTSYCL_GCC_TOOLCHAIN}/lib/gcc/*/*")
find_library(PTSYCL_GOMP_LIBRARY NAMES gomp
    HINTS "${PTSYCL_GCC_TOOLCHAIN}/lib64" "${PTSYCL_GCC_TOOLCHAIN}/lib" ${_gcc_lib_dirs})
if(NOT PTSYCL_GOMP_LIBRARY)
    message(FATAL_ERROR "libgomp not found under PTSYCL_GCC_TOOLCHAIN=${PTSYCL_GCC_TOOLCHAIN}")
endif()

target_link_libraries(pjrt_paras PRIVATE
    "${PTSYCL_CLAP_LIBRARY}"
    "${PTSYCL_GOMP_LIBRARY}"
    pthread
    dl
)

if(PARAS_PJRT_GPU_RUNTIME_LIB)
    target_link_libraries(pjrt_paras PRIVATE "${PARAS_PJRT_GPU_RUNTIME_LIB}")
endif()

if(PARAS_PJRT_BACKEND_DEFINE STREQUAL "PARAS_PJRT_BACKEND_SYCL")
    target_link_options(pjrt_paras PRIVATE ${PTSYCL_DEVICE_FLAGS})
endif()

target_link_options(pjrt_paras PRIVATE
    --gcc-toolchain=${PTSYCL_GCC_TOOLCHAIN}
)

set(_pjrt_rpaths
    "${PARAS_HOME}/lib"
    "${PTSYCL_GCC_TOOLCHAIN}/lib64"
    "${PTSYCL_CLAP_LIBRARY_DIR}"
)
if(PTSYCL_PARAS_RUNTIME_DIR)
    list(INSERT _pjrt_rpaths 0 "${PTSYCL_PARAS_RUNTIME_DIR}")
endif()
if(PARAS_PJRT_GPU_RUNTIME_LIB)
    get_filename_component(_gpu_rt_dir "${PARAS_PJRT_GPU_RUNTIME_LIB}" DIRECTORY)
    list(APPEND _pjrt_rpaths "${_gpu_rt_dir}")
endif()

list(JOIN _pjrt_rpaths ";" _pjrt_rpath_str)
set_target_properties(pjrt_paras PROPERTIES
    BUILD_RPATH "${_pjrt_rpath_str}"
    INSTALL_RPATH "${_pjrt_rpath_str}"
)
