#include "dot_product_v2.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>


template <typename T>
class DotProductV2Test : public ::testing::TestWithParam<int> {
protected:
    void RunAndCheck() {
        const int N = GetParam();

        T* a = nullptr;
        T* b = nullptr;
        CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(T)));
        CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(T)));

        for (int i = 0; i < N; i++) {
            a[i] = static_cast<T>(i + 1);
            b[i] = static_cast<T>(N - i);
        }

        if (N > 0) {
            CUDA_CHECK(cudaMemPrefetchAsync(a, N * sizeof(T), 0, 0));
            CUDA_CHECK(cudaMemPrefetchAsync(b, N * sizeof(T), 0, 0));
        }

        T expected = static_cast<T>(0);
        for (int i = 0; i < N; i++) {
            expected += a[i] * b[i];
        }

        T result = dot_product(N, a, b);

        EXPECT_EQ(result, expected);

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
    }
};

// gtest has no single macro combining type- and value-parameterization, so
// each type needs its own concrete fixture alias + TEST_P + INSTANTIATE.
#define DOT_PRODUCT_V2_TEST_FOR_TYPE(TypeName, Type)                        \
    using DotProductV2Test##TypeName = DotProductV2Test<Type>;              \
    TEST_P(DotProductV2Test##TypeName, MatchesCpuDotProduct) { RunAndCheck(); } \
    INSTANTIATE_TEST_SUITE_P(                                               \
        VariousN,                                                           \
        DotProductV2Test##TypeName,                                        \
        ::testing::Values(0, 1, 3, 255, 256, 257, 1000),                    \
        [](const ::testing::TestParamInfo<int>& info) {                    \
            return "N" + std::to_string(info.param);                       \
        }                                                                   \
    )

DOT_PRODUCT_V2_TEST_FOR_TYPE(Int, int);
// DOT_PRODUCT_V2_TEST_FOR_TYPE(Float, float);
// DOT_PRODUCT_V2_TEST_FOR_TYPE(Double, double);
// DOT_PRODUCT_V2_TEST_FOR_TYPE(UnsignedLongLong, unsigned long long);
