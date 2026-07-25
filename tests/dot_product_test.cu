#include "dot_product.cuh"
#include "cuda_check.cuh"

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

// Managed memory is host-accessible, so tests can fill/read x and y directly
// without any explicit copy.
void allocManaged(int N, float **x, float **y)
{
	CUDA_CHECK(cudaMallocManaged(x, N * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(y, N * sizeof(float)));
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
	float *x, *y;
	allocManaged(1, &x, &y);

	EXPECT_FLOAT_EQ(dotProductGPU(0, x, y), 0.f);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
}

TEST(DotProduct, SingleElement)
{
	float *x, *y;
	allocManaged(1, &x, &y);
	x[0] = 3.f;
	y[0] = 5.f;

	expectClose(15.f, dotProductGPU(1, x, y));

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
}

TEST(DotProduct, FewerElementsThanThreadsPerBlock)
{
	int N = 50; // < THREADS_PER_BLOCK (128)
	float *x, *y;
	allocManaged(N, &x, &y);
	fillRandom(N, x, y, 42);

	expectClose(dotProductCPU(N, x, y), dotProductGPU(N, x, y));

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
}

TEST(DotProduct, ElementCountNotMultipleOfGridStride)
{
	int N = 100003; // deliberately not a multiple of THREADS_PER_BLOCK * NUM_BLOCKS
	float *x, *y;
	allocManaged(N, &x, &y);
	fillRandom(N, x, y, 7);

	expectClose(dotProductCPU(N, x, y), dotProductGPU(N, x, y));

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
}

TEST(DotProduct, LargeRandomInput)
{
	int N = 1 << 20;
	float *x, *y;
	allocManaged(N, &x, &y);
	fillRandom(N, x, y, 123);

	expectClose(dotProductCPU(N, x, y), dotProductGPU(N, x, y));

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
}
