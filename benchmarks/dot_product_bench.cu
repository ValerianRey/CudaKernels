#include "dot_product.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>
#include <iostream>

int main(void)
{
	int N = 1 << 20;
	int warmupIters = 5;
	int timedIters = 50;

	CudaManagedArray<float> x(N);
	CudaManagedArray<float> y(N);
	for (int i = 0; i < N; i++) {
		x[i] = 3.0f;
		y[i] = 5.0f;
	}
	x.prefetchToDevice(0, 0);
	y.prefetchToDevice(0, 0);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Warm up: the first launch pays for context init / JIT, and shouldn't
	// be counted towards the measured time.
	for (int i = 0; i < warmupIters; i++) {
		dotProductGPU(N, x.get(), y.get());
	}

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < timedIters; i++) {
		dotProductGPU(N, x.get(), y.get());
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float totalMs = 0.f;
	CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
	float avgMs = totalMs / timedIters;

	// dotProduct reads x and y once each; the (small, constant-size) block
	// results buffer is not counted.
	double bytesPerCall = 2.0 * N * sizeof(float);
	double gbPerSec = (bytesPerCall / 1e9) / (avgMs / 1e3);

	std::cout << "N: " << N << "\n";
	std::cout << "Average kernel time: " << avgMs << " ms\n";
	std::cout << "Achieved bandwidth: " << gbPerSec << " GB/s\n";

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}
