// Based on the excellent tutorial https://developer.nvidia.com/blog/even-easier-introduction-cuda/

#include "add_vector.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void add(int N, float *sum, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < N; i += stride)
		sum[i] = x[i] + y[i];
}

void addVectorsGPU(int N, float *x, float *y, float *sum)
{
	if (N <= 0)
		return;

	int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	add<<<numBlocks, BLOCK_SIZE>>>(N, sum, x, y);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}
