#include "add_vector.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(AddVector, ZeroElements)
{	
	CudaManagedArray<float> x(1);
	CudaManagedArray<float> y(1);
	CudaManagedArray<float> sum(1);

	addVectorsGPU(0, x.get(), y.get(), sum.get());
}

TEST(AddVector, SingleElement)
{
	CudaManagedArray<float> x(1);
	CudaManagedArray<float> y(1);
	CudaManagedArray<float> sum(1);

	x[0] = 3.f;
	y[0] = 5.f;

	addVectorsGPU(1, x.get(), y.get(), sum.get());

	EXPECT_FLOAT_EQ(sum[0], 8.f);
}

TEST(AddVector, FewerElementsThanBlockSize)
{
	int N = 50; // < BLOCK_SIZE (256)
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	CudaManagedArray<float> sum(N);
	for (int i = 0; i < N; i++) {
		x[i] = static_cast<float>(i);
		y[i] = static_cast<float>(2 * i);
	}

	addVectorsGPU(N, x.get(), y.get(), sum.get());

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.f * i);
}

TEST(AddVector, ElementCountNotMultipleOfGridStride)
{
	int N = 100003; // deliberately not a multiple of BLOCK_SIZE
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	CudaManagedArray<float> sum(N);

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	addVectorsGPU(N, x.get(), y.get(), sum.get());

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.0f);
}

TEST(AddVector, LargeInput)
{
	int N = 1 << 20;
	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	CudaManagedArray<float> sum(N);

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	addVectorsGPU(N, x.get(), y.get(), sum.get());

	for (int i = 0; i < N; i++)
		EXPECT_FLOAT_EQ(sum[i], 3.0f);
}
