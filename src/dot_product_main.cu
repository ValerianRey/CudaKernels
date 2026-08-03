#include "dot_product.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main(void)
{
	int N = 1<<20;

	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);

	for (int i = 0; i < N; i++) {
		x[i] = 3.0f;
		y[i] = 5.0f;
	}

	// Prefetch to the GPU
	x.prefetchToDevice(0, 0);
	y.prefetchToDevice(0, 0);

	float result = dotProductGPU(N, x.get(), y.get());

	float expectedResult = N * 15.0f;
	float error = expectedResult - result;

	std::cout<<"Error: "<<error<<std::endl;

	return 0;
}
