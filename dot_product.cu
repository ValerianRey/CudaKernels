// Heavily inspired from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS 256

__device__ void reduce(float *block_results)
{
	//Reduce block_results such that at the end, block_results[0] = sum(block_results)
	if (threadIdx.x < 64)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 64];
	__syncthreads();

	if (threadIdx.x < 32)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 32];
	__syncthreads();

	if (threadIdx.x < 16)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 16];
	__syncthreads();

	if (threadIdx.x < 8)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 8];
	__syncthreads();
	
	if (threadIdx.x < 4)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 4];
	__syncthreads();
	
	if (threadIdx.x < 2)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 2];
	__syncthreads();

	if (threadIdx.x < 1)
		block_results[threadIdx.x] = block_results[threadIdx.x] + block_results[threadIdx.x + 1];
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

int main(void)
{
	int N = 1<<20;

	float *x, *y, *results;

	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&results, NUM_BLOCKS*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 3.0f;
		y[i] = 5.0f;
	}

	// Prefetch to the GPU
	cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(results, NUM_BLOCKS*sizeof(float), 0, 0);

	dotProduct<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(N, x, y, results);
	cudaDeviceSynchronize();
	cudaFree(x);
	cudaFree(y);
	
	float result = 0.f;
	
	// Final reduction over the blocks
	for (int i = 0; i < NUM_BLOCKS; i++) {
		result += results[i];
	}
	
	cudaFree(results);

	float expectedResult = N * 15.0f;
	float error = expectedResult - result;

	std::cout<<"Error: "<<error<<std::endl;
	
	return 0;
}
