#include "softmax_2d.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>


namespace {

// A parallel tree reduction sums in a different order than the sequential
// CPU reference, and the host/device exp() implementations aren't bit
// identical either, so results only match up to floating-point rounding.
void expectClose(float expected, float actual) {
    float tolerance = std::max(1e-4F, std::fabs(expected) * 1e-4F);
    EXPECT_NEAR(expected, actual, tolerance);
}

} // namespace

class Softmax2dTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(Softmax2dTest, MatchesCpuSoftmax) {
    const int rows = GetParam().first;
    const int cols = GetParam().second;
    const int size = rows * cols;

    float* input = nullptr;
    CUDA_CHECK(cudaMallocManaged(&input, size * sizeof(float)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0F, 10.0F);
    std::vector<float> original(size);
    for (int i = 0; i < size; i++) {
        original[i] = dist(rng);
        input[i] = original[i];
    }

    CUDA_CHECK(cudaMemPrefetchAsync(input, size * sizeof(float), 0, 0));

    softmax_2d(input, rows, cols);

    for (int r = 0; r < rows; r++) {
        float row_max = std::numeric_limits<float>::lowest();
        for (int c = 0; c < cols; c++) {
            row_max = std::max(row_max, original[r * cols + c]);
        }

        float denom = 0.F;
        std::vector<float> exps(cols);
        for (int c = 0; c < cols; c++) {
            exps[c] = std::exp(original[r * cols + c] - row_max);
            denom += exps[c];
        }

        for (int c = 0; c < cols; c++) {
            expectClose(exps[c] / denom, input[r * cols + c]);
        }
    }

    CUDA_CHECK(cudaFree(input));
}

INSTANTIATE_TEST_SUITE_P(
    VariousShapes,
    Softmax2dTest,
    ::testing::Values(
        std::pair<int, int>{1, 1},
        std::pair<int, int>{1, 3},
        std::pair<int, int>{3, 1},
        std::pair<int, int>{5, 5},
        std::pair<int, int>{5, 255},
        std::pair<int, int>{5, 256},
        std::pair<int, int>{5, 257},
        std::pair<int, int>{2, 1000},
        std::pair<int, int>{300, 3}
    ),
    [](const ::testing::TestParamInfo<std::pair<int, int>>& info) {
        return "R" + std::to_string(info.param.first)
             + "C" + std::to_string(info.param.second);
    }
);
