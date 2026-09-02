cmake_minimum_required(VERSION 3.24)
project(CudaKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- dependencies (fetched automatically on first configure) ---

include(FetchContent)

FetchContent_Declare(
	cutlass
	GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
	GIT_TAG v3.9.2
	GIT_SHALLOW TRUE
)
# CUTLASS is used header-only here (gemm_cutlass.cu just includes its device::Gemm
# API), so only fetch its source; don't add_subdirectory and build its own
# examples/tests/profiler via its own (much heavier) CMake project.
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
	FetchContent_Populate(cutlass)
endif()

FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.15.2
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- kernel registry ---
# List a kernel's name once, here, to build it into the library and get
# tests/<kernel>_test.cu picked up as a test. Its source must be
# src/<kernel>.cu, or src/<kernel>.cuh for a header-only template kernel.
# Additionally list it under KERNELS_WITH_DEMO or KERNELS_WITH_BENCH to also
# build src/<kernel>_main.cu as a demo or benchmarks/<kernel>_bench.cu as a
# benchmark.

set(KERNELS
	add_vector
	dot_product
	matrix_product
	gemm
	gemm_cutlass
	softmax
	reductions
	dot_product_v2
	histogram
	transpose
	softmax_2d
)

set(KERNELS_WITH_DEMO
	add_vector
	dot_product
	matrix_product
	gemm
	gemm_cutlass
)

set(KERNELS_WITH_BENCH
	dot_product
	gemm
	gemm_cutlass
	softmax
)

# --- kernel library, shared by every demo, benchmark and test ---

set(KERNEL_SOURCES)
foreach(kernel ${KERNELS})
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel}.cu)
		list(APPEND KERNEL_SOURCES src/${kernel}.cu)
	else()
		list(APPEND KERNEL_SOURCES src/${kernel}.cuh)
	endif()
endforeach()

add_library(cuda_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(cuda_kernels PUBLIC src)
target_include_directories(cuda_kernels PRIVATE ${cutlass_SOURCE_DIR}/include)
# PUBLIC: several kernels (e.g. reductions.cuh, dot_product_v2.cuh) are
# header-only templates, so their device code is actually compiled wherever
# they're included and instantiated (demos, benchmarks, tests), not just in
# this library's own .cu files. These flags need to reach those TUs too.
#
# CUTLASS headers use standard-library constexpr helpers from host/device
# functions (e.g. std::get on cutlass::Coord), which needs this relaxation.
target_compile_options(cuda_kernels PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# -G gives full device-side debug info for cuda-gdb (breakpoints/stepping in
# kernels) but disables device-code optimizations, so only enable it in Debug.
target_compile_options(cuda_kernels PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G>>)
# -lineinfo maps profiler samples (Nsight Systems/Compute) back to source
# lines without disabling optimizations, so keep it on outside of Debug.
target_compile_options(cuda_kernels PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:$<$<NOT:$<CONFIG:Debug>>:-lineinfo>>)

# --- demos ---

foreach(kernel ${KERNELS_WITH_DEMO})
	add_executable(${kernel}_demo src/${kernel}_main.cu)
	target_link_libraries(${kernel}_demo PRIVATE cuda_kernels)
endforeach()

list(TRANSFORM KERNELS_WITH_DEMO APPEND "_demo" OUTPUT_VARIABLE DEMO_TARGETS)
add_custom_target(demo DEPENDS ${DEMO_TARGETS})

# --- benchmarks ---

foreach(kernel ${KERNELS_WITH_BENCH})
	add_executable(${kernel}_bench benchmarks/${kernel}_bench.cu)
	target_link_libraries(${kernel}_bench PRIVATE cuda_kernels)
endforeach()

list(TRANSFORM KERNELS_WITH_BENCH APPEND "_bench" OUTPUT_VARIABLE BENCH_TARGETS)
add_custom_target(bench DEPENDS ${BENCH_TARGETS})

# --- tests ---

enable_testing()

foreach(kernel ${KERNELS})
	add_executable(${kernel}_test tests/${kernel}_test.cu)
	target_link_libraries(${kernel}_test PRIVATE cuda_kernels gtest_main)
	add_test(NAME ${kernel} COMMAND ${kernel}_test)
endforeach()
