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

# --- kernel library, shared by every demo, benchmark and test ---

add_library(cuda_kernels STATIC
	src/add_vector.cu
	src/dot_product.cu
	src/matrix_product.cu
	src/gemm.cu
	src/gemm_cutlass.cu
	src/softmax.cu
)
target_include_directories(cuda_kernels PUBLIC src)
target_include_directories(cuda_kernels PRIVATE ${cutlass_SOURCE_DIR}/include)
# CUTLASS headers use standard-library constexpr helpers from host/device
# functions (e.g. std::get on cutlass::Coord), which needs this relaxation.
target_compile_options(cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# -G gives full device-side debug info for cuda-gdb (breakpoints/stepping in
# kernels) but disables device-code optimizations, so only enable it in Debug.
target_compile_options(cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G>>)

# --- demos ---

foreach(kernel add_vector dot_product matrix_product gemm gemm_cutlass)
	add_executable(${kernel}_demo src/${kernel}_main.cu)
	target_link_libraries(${kernel}_demo PRIVATE cuda_kernels)
endforeach()

add_custom_target(demo DEPENDS add_vector_demo dot_product_demo matrix_product_demo gemm_demo gemm_cutlass_demo)

# --- benchmarks ---

add_executable(dot_product_bench benchmarks/dot_product_bench.cu)
target_link_libraries(dot_product_bench PRIVATE cuda_kernels)

add_executable(gemm_bench benchmarks/gemm_bench.cu)
target_link_libraries(gemm_bench PRIVATE cuda_kernels)

add_executable(gemm_cutlass_bench benchmarks/gemm_cutlass_bench.cu)
target_link_libraries(gemm_cutlass_bench PRIVATE cuda_kernels)

add_custom_target(bench DEPENDS dot_product_bench gemm_bench gemm_cutlass_bench)

# --- tests ---

enable_testing()

foreach(kernel add_vector dot_product matrix_product gemm gemm_cutlass softmax)
	add_executable(${kernel}_test tests/${kernel}_test.cu)
	target_link_libraries(${kernel}_test PRIVATE cuda_kernels gtest_main)
	add_test(NAME ${kernel} COMMAND ${kernel}_test)
endforeach()
