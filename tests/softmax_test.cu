#include "softmax.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

void softmax_cpu(int N, float* input, float* out) {
    float denom = 0.F;
    for (int i = 0; i < N; i++) {
        denom += exp(input[i]);
    }
    for (int i = 0; i < N; i++) {
        out[i] = exp(input[i]) / denom;
    }
}

TEST(Softmax, Small) {
    constexpr int N = 3;
    CudaManagedArray<float> x(N);
    x[0] = 1.F;
    x[1] = 2.F;
    x[2] = -5.F;

    float expected[N];
    softmax_cpu(N, x.get(), expected);

    softmax_(N, x.get());

    for (int i = 0; i < x.size(); i++) {
        EXPECT_FLOAT_EQ(x[i], expected[i]);
    }
}

TEST(Softmax, Large) {
    constexpr int N = 1000;
    CudaManagedArray<float> x(N);
    for (int i = 0; i < N; i++) {
        x[i] =  1.F / ((float) i + 1.F);
    }

    float expected[N];
    softmax_cpu(N, x.get(), expected);

    softmax_(N, x.get());

    for (int i = 0; i < x.size(); i++) {
        EXPECT_FLOAT_EQ(x[i], expected[i]);
    }
}