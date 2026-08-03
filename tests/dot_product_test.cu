#include "dot_product.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>

namespace {

float dotProductCPU(int N, const float *x, const float *y)
{
	float result = 0.f;
	for (int i = 0; i < N; i++) {
		result += x[i] * y[i];
	}
	return result;
}

void fillRandom(int N, float *x, float *y, unsigned seed)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	for (int i = 0; i < N; i++) {
		x[i] = dist(rng);
		y[i] = dist(rng);
	}
}

// A parallel tree reduction sums in a different order than the sequential
// CPU reference, so results only match up to floating-point rounding.
void expectClose(float expected, float actual)
{
	float tolerance = std::max(1e-4f, std::fabs(expected) * 1e-4f);
	EXPECT_NEAR(expected, actual, tolerance);
}

} // namespace

TEST(DotProduct, ZeroElements)
{
	CudaManagedArray<float> x(1);
	CudaManagedArray<float> y(1);

	EXPECT_FLOAT_EQ(dotProductGPU(0, x.get(), y.get()), 0.f);
}

TEST(DotProduct, SingleElement)
{
	CudaManagedArray<float> x(1);
	CudaManagedArray<float> y(1);
	x[0] = 3.f;
	y[0] = 5.f;

	expectClose(15.f, dotProductGPU(1, x.get(), y.get()));
}

TEST(DotProduct, FewerElementsThanThreadsPerBlock)
{
	int N = 50; // < THREADS_PER_BLOCK (128)
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	fillRandom(N, x.get(), y.get(), 42);

	expectClose(dotProductCPU(N, x.get(), y.get()), dotProductGPU(N, x.get(), y.get()));
}

TEST(DotProduct, ElementCountNotMultipleOfGridStride)
{
	int N = 100003; // deliberately not a multiple of THREADS_PER_BLOCK * NUM_BLOCKS
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	fillRandom(N, x.get(), y.get(), 7);

	expectClose(dotProductCPU(N, x.get(), y.get()), dotProductGPU(N, x.get(), y.get()));
}

TEST(DotProduct, LargeRandomInput)
{
	int N = 1 << 20;
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	fillRandom(N, x.get(), y.get(), 123);

	expectClose(dotProductCPU(N, x.get(), y.get()), dotProductGPU(N, x.get(), y.get()));
}
