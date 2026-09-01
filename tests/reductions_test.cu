#include "reductions.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>


template <typename T>
class SumTest : public ::testing::TestWithParam<int> {
protected:
    void RunAndCheck() {
        const int N = GetParam();

        T* input = nullptr;
        CUDA_CHECK(cudaMallocManaged(&input, N * sizeof(T)));

        for (int i = 0; i < N; i++) {
            input[i] = static_cast<T>(i + 1);
        }

        if (N > 0) {
            CUDA_CHECK(cudaMemPrefetchAsync(input, N * sizeof(T), 0, 0));
        }

        T expected = static_cast<T>(0);
        for (int i = 0; i < N; i++) {
            expected += input[i];
        }

        T s = sum_reduction(N, input);

        EXPECT_EQ(s, expected);

        CUDA_CHECK(cudaFree(input));
    }
};

// gtest has no single macro combining type- and value-parameterization, so
// each type needs its own concrete fixture alias + TEST_P + INSTANTIATE.
#define SUM_TEST_FOR_TYPE(TypeName, Type)                                   \
    using SumTest##TypeName = SumTest<Type>;                                \
    TEST_P(SumTest##TypeName, MatchesCpuSum) { RunAndCheck(); }             \
    INSTANTIATE_TEST_SUITE_P(                                               \
        VariousN,                                                           \
        SumTest##TypeName,                                                  \
        ::testing::Values(0, 1, 3, 255, 256, 257, 1000),                    \
        [](const ::testing::TestParamInfo<int>& info) {                    \
            return "N" + std::to_string(info.param);                       \
        }                                                                   \
    )

SUM_TEST_FOR_TYPE(Int, int);
SUM_TEST_FOR_TYPE(Float, float);
SUM_TEST_FOR_TYPE(Double, double);
SUM_TEST_FOR_TYPE(UnsignedLongLong, unsigned long long);


template <typename T>
class MaxTest : public ::testing::TestWithParam<int> {
protected:
    void RunAndCheck() {
        const int N = GetParam();

        T* input = nullptr;
        CUDA_CHECK(cudaMallocManaged(&input, N * sizeof(T)));

        for (int i = 0; i < N; i++) {
            input[i] = static_cast<T>(i + 1);
        }

        if (N > 0) {
            CUDA_CHECK(cudaMemPrefetchAsync(input, N * sizeof(T), 0, 0));
        }

        T expected = std::numeric_limits<T>::lowest();
        for (int i = 0; i < N; i++) {
            expected = std::max(expected, input[i]);
        }

        T m = max_reduction(N, input);

        EXPECT_EQ(m, expected);

        CUDA_CHECK(cudaFree(input));
    }
};

#define MAX_TEST_FOR_TYPE(TypeName, Type)                                   \
    using MaxTest##TypeName = MaxTest<Type>;                                \
    TEST_P(MaxTest##TypeName, MatchesCpuMax) { RunAndCheck(); }             \
    INSTANTIATE_TEST_SUITE_P(                                               \
        VariousN,                                                           \
        MaxTest##TypeName,                                                  \
        ::testing::Values(0, 1, 3, 255, 256, 257, 1000),                    \
        [](const ::testing::TestParamInfo<int>& info) {                    \
            return "N" + std::to_string(info.param);                       \
        }                                                                   \
    )

MAX_TEST_FOR_TYPE(Int, int);
MAX_TEST_FOR_TYPE(UnsignedLongLong, unsigned long long);
