// Heavily inspired from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu

#include "dot_product.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS 256

__device__ void reduce(float *block_results)
{
	// Reduce block_results such that at the end, block_results[0] = sum(block_results)
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride)
			block_results[threadIdx.x] += block_results[threadIdx.x + stride];
		__syncthreads();
	}
}

__global__ void dotProduct(int N, float *x, float *y, float *results)
{
	float result = 0.f;

	int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ float block_results[THREADS_PER_BLOCK];

	for(int i = startIndex; i < N; i += stride) {
		result += x[i] * y[i];
	}

	block_results[threadIdx.x] = result;
	__syncthreads();

	reduce(block_results);

	if (threadIdx.x == 0)
		results[blockIdx.x] = block_results[0];
}

float dotProductGPU(int N, float *x, float *y)
{
	float *results;
	CUDA_CHECK(cudaMallocManaged(&results, NUM_BLOCKS * sizeof(float)));

	dotProduct<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(N, x, y, results);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	float result = 0.f;
	for (int i = 0; i < NUM_BLOCKS; i++) {
		result += results[i];
	}

	CUDA_CHECK(cudaFree(results));

	return result;
}
