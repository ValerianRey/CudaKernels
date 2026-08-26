cmake_minimum_required(VERSION 3.24)
project(CudaKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- kernel library, shared by every demo, benchmark and test ---

add_library(cuda_kernels STATIC
	src/add_vector.cu
	src/dot_product.cu
	src/matrix_product.cu
)
target_include_directories(cuda_kernels PUBLIC src)

# --- demos ---

foreach(kernel add_vector dot_product matrix_product)
	add_executable(${kernel}_demo src/${kernel}_main.cu)
	target_link_libraries(${kernel}_demo PRIVATE cuda_kernels)
endforeach()

add_custom_target(demo DEPENDS add_vector_demo dot_product_demo matrix_product_demo)

# --- benchmark ---

add_executable(dot_product_bench benchmarks/dot_product_bench.cu)
target_link_libraries(dot_product_bench PRIVATE cuda_kernels)

add_custom_target(bench DEPENDS dot_product_bench)

# --- tests (GoogleTest is fetched automatically on first configure) ---

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.15.2
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

foreach(kernel add_vector dot_product matrix_product)
	add_executable(${kernel}_test tests/${kernel}_test.cu)
	target_link_libraries(${kernel}_test PRIVATE cuda_kernels gtest_main)
	add_test(NAME ${kernel} COMMAND ${kernel}_test)
endforeach()
