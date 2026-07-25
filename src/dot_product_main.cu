#include "dot_product.cuh"
#include "cuda_check.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main(void)
{
	int N = 1<<20;

	float *x, *y;

	CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&y, N*sizeof(float)));

	for (int i = 0; i < N; i++) {
		x[i] = 3.0f;
		y[i] = 5.0f;
	}

	// Prefetch to the GPU
	CUDA_CHECK(cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0));

	float result = dotProductGPU(N, x, y);

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));

	float expectedResult = N * 15.0f;
	float error = expectedResult - result;

	std::cout<<"Error: "<<error<<std::endl;

	return 0;
}
