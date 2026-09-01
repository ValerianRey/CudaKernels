#include "transpose.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <utility>


// Expected kernel interface (implement in src/transpose.cuh):
//   template <typename T>
//   void transpose(T* input, T* output, int rows, int cols);
// Transposes a rows x cols row-major matrix `input` into a cols x rows
// row-major matrix `output`.

template <typename T>
class TransposeTest : public ::testing::TestWithParam<std::pair<int, int>> {
protected:
    void RunAndCheck() {
        const int rows = GetParam().first;
        const int cols = GetParam().second;
        const int size = rows * cols;

        T* input = nullptr;
        T* output = nullptr;
        CUDA_CHECK(cudaMallocManaged(&input, size * sizeof(T)));
        CUDA_CHECK(cudaMallocManaged(&output, size * sizeof(T)));

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                input[r * cols + c] = static_cast<T>(r * cols + c);
            }
        }

        CUDA_CHECK(cudaMemPrefetchAsync(input, size * sizeof(T), 0, 0));

        transpose(input, output, rows, cols);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                EXPECT_EQ(output[c * rows + r], input[r * cols + c])
                    << "mismatch at input row " << r << ", col " << c;
            }
        }

        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
    }
};

// gtest has no single macro combining type- and value-parameterization, so
// each type needs its own concrete fixture alias + TEST_P + INSTANTIATE.
#define TRANSPOSE_TEST_FOR_TYPE(TypeName, Type)                             \
    using TransposeTest##TypeName = TransposeTest<Type>;                    \
    TEST_P(TransposeTest##TypeName, MatchesCpuTranspose) { RunAndCheck(); } \
    INSTANTIATE_TEST_SUITE_P(                                               \
        VariousShapes,                                                     \
        TransposeTest##TypeName,                                           \
        ::testing::Values(                                                 \
            std::pair<int, int>{1, 1},                                     \
            std::pair<int, int>{1, 7},                                     \
            std::pair<int, int>{7, 1},                                     \
            std::pair<int, int>{5, 5},                                     \
            std::pair<int, int>{16, 16},                                   \
            std::pair<int, int>{17, 15},                                   \
            std::pair<int, int>{32, 33},                                   \
            std::pair<int, int>{255, 257},                                 \
            std::pair<int, int>{256, 256},                                 \
            std::pair<int, int>{257, 255},                                 \
            std::pair<int, int>{300, 300}                                  \
        ),                                                                  \
        [](const ::testing::TestParamInfo<std::pair<int, int>>& info) {    \
            return "R" + std::to_string(info.param.first)                  \
                 + "C" + std::to_string(info.param.second);                \
        }                                                                   \
    )

TRANSPOSE_TEST_FOR_TYPE(Int, int);
TRANSPOSE_TEST_FOR_TYPE(Float, float);
TRANSPOSE_TEST_FOR_TYPE(Double, double);
TRANSPOSE_TEST_FOR_TYPE(UnsignedLongLong, unsigned long long);
