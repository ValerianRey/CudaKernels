#include "histogram.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>


constexpr int NUM_BINS = 4;

template <typename T>
class HistogramTest : public ::testing::TestWithParam<int> {
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

        std::vector<int> expected(NUM_BINS, 0);
        if (N > 0) {
            T min_val = input[0];
            T max_val = input[0];
            for (int i = 1; i < N; i++) {
                min_val = std::min(min_val, input[i]);
                max_val = std::max(max_val, input[i]);
            }

            T bin_size = (max_val - min_val + static_cast<T>(NUM_BINS) - static_cast<T>(1)) / static_cast<T>(NUM_BINS);
            if (bin_size == static_cast<T>(0)) {
                bin_size = static_cast<T>(1);
            }
            for (int i = 0; i < N; i++) {
                int bin_id = static_cast<int>((input[i] - min_val) / bin_size);
                if (bin_id >= NUM_BINS) {
                    bin_id = NUM_BINS - 1;
                }
                expected[bin_id]++;
            }
        }

        int* result = nullptr;
        CUDA_CHECK(cudaMallocManaged(&result, NUM_BINS * sizeof(int)));

        histogram(N, input, NUM_BINS, result);

        for (int i = 0; i < NUM_BINS; i++) {
            EXPECT_EQ(result[i], expected[i]);
        }

        CUDA_CHECK(cudaFree(result));
        CUDA_CHECK(cudaFree(input));
    }
};

// gtest has no single macro combining type- and value-parameterization, so
// each type needs its own concrete fixture alias + TEST_P + INSTANTIATE.
#define HISTOGRAM_TEST_FOR_TYPE(TypeName, Type)                             \
    using HistogramTest##TypeName = HistogramTest<Type>;                    \
    TEST_P(HistogramTest##TypeName, MatchesCpuHistogram) { RunAndCheck(); } \
    INSTANTIATE_TEST_SUITE_P(                                               \
        VariousN,                                                           \
        HistogramTest##TypeName,                                           \
        ::testing::Values(0, 1, 3, 255, 256, 257, 1000),                    \
        [](const ::testing::TestParamInfo<int>& info) {                    \
            return "N" + std::to_string(info.param);                       \
        }                                                                   \
    )

HISTOGRAM_TEST_FOR_TYPE(Int, int);
// HISTOGRAM_TEST_FOR_TYPE(Float, float);
// HISTOGRAM_TEST_FOR_TYPE(Double, double);
HISTOGRAM_TEST_FOR_TYPE(UnsignedLongLong, unsigned long long);
