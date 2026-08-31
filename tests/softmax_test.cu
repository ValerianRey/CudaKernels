#include "softmax.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(Softmax, Simple) {
    constexpr int N = 3;
    CudaManagedArray<float> x(N);
    x[0] = 1.F;
    x[1] = 2.F;
    x[2] = -5.F;

    float expected[N];
    float denom = 0.F;
    for (int i = 0; i < x.size(); i++) {
        denom += exp(x[i]);
    }
    for (int i = 0; i < x.size(); i++) {
        expected[i] = exp(x[i]) / denom;
    }

    softmax_(N, x.get());

    for (int i = 0; i < x.size(); i++) {
        EXPECT_FLOAT_EQ(x[i], expected[i]);
    }
}