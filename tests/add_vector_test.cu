#include "add_vector.cuh"
#include "cuda_check.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

namespace {

// Managed memory is host-accessible, so tests can fill/read x, y and sum
// directly without any explicit copy.
void allocManaged(int N, float **x, float **y, float **sum)
{
	CUDA_CHECK(cudaMallocManaged(x, N * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(y, N * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(sum, N * sizeof(float)));
}

} // namespace

TEST(AddVector, ZeroElements)
{
	float *x, *y, *sum;
	allocManaged(1, &x, &y, &sum);

	addVectorsGPU(0, x, y, sum);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));
}

TEST(AddVector, SingleElement)
{
	float *x, *y, *sum;
	allocManaged(1, &x, &y, &sum);
	x[0] = 3.f;
	y[0] = 5.f;

	addVectorsGPU(1, x, y, sum);

	EXPECT_FLOAT_EQ(sum[0], 8.f);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));
}

TEST(AddVector, FewerElementsThanBlockSize)
{
	int N = 50; // < BLOCK_SIZE (256)
	float *x, *y, *sum;
	allocManaged(N, &x, &y, &sum);
	for (int i = 0; i < N; i++) {
		x[i] = static_cast<float>(i);
		y[i] = static_cast<float>(2 * i);
	}

	addVectorsGPU(N, x, y, sum);

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.f * i);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));
}

TEST(AddVector, ElementCountNotMultipleOfGridStride)
{
	int N = 100003; // deliberately not a multiple of BLOCK_SIZE
	float *x, *y, *sum;
	allocManaged(N, &x, &y, &sum);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	addVectorsGPU(N, x, y, sum);

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.0f);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));
}

TEST(AddVector, LargeInput)
{
	int N = 1 << 20;
	float *x, *y, *sum;
	allocManaged(N, &x, &y, &sum);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	addVectorsGPU(N, x, y, sum);

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.0f);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));
}
