#include "add_vector.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

int main(void)
{
	int N = 1<<20;

	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	CudaManagedArray<float> sum(N);

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	x.prefetchToDevice(0, 0);
	y.prefetchToDevice(0, 0);
	sum.prefetchToDevice(0, 0);

	addVectorsGPU(N, x.get(), y.get(), sum.get());

	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(sum[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	return 0;
}
