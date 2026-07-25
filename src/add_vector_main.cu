#include "add_vector.cuh"
#include "cuda_check.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

int main(void)
{
	int N = 1<<20;

	float *x, *y, *sum;
	CUDA_CHECK(cudaMallocManaged(&x, N*sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&y, N*sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&sum, N*sizeof(float)));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Prefetch to the GPU
	CUDA_CHECK(cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(sum, N*sizeof(float), 0, 0));

	addVectorsGPU(N, x, y, sum);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(sum[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(sum));

	return 0;
}
