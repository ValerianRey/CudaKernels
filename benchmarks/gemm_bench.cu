#include "gemm.cuh"
#include "matrix_product.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>
#include <iostream>

int main(void)
{
	int m = 2048, n = 2048, p = 2048;
	int warmupIters = 3;
	int timedIters = 10;

	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> naiveC(m * p);
	CudaManagedArray<float> gemmC(m * p);

	for (int i = 0; i < m * n; i++)
		A[i] = static_cast<float>((i % 7) - 3) * 0.1f;
	for (int i = 0; i < n * p; i++)
		B[i] = static_cast<float>((i % 5) - 2) * 0.1f;

	A.prefetchToDevice(0, 0);
	B.prefetchToDevice(0, 0);
	naiveC.prefetchToDevice(0, 0);
	gemmC.prefetchToDevice(0, 0);
	CUDA_CHECK(cudaDeviceSynchronize());

	double flops = 2.0 * m * n * p; // one multiply + one add per inner-product term

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// --- naive matrix_product_GPU (one thread per output element) ---

	for (int i = 0; i < warmupIters; i++)
		matrix_product_GPU(m, n, p, A.get(), B.get(), naiveC.get());

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < timedIters; i++)
		matrix_product_GPU(m, n, p, A.get(), B.get(), naiveC.get());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float naiveMs = 0.f;
	CUDA_CHECK(cudaEventElapsedTime(&naiveMs, start, stop));
	naiveMs /= timedIters;

	// --- tiled + register-blocked gemmGPU (alpha=1, beta=0 == plain A@B) ---

	for (int i = 0; i < warmupIters; i++)
		gemmGPU(m, n, p, 1.f, A.get(), B.get(), 0.f, gemmC.get());

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < timedIters; i++)
		gemmGPU(m, n, p, 1.f, A.get(), B.get(), 0.f, gemmC.get());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float gemmMs = 0.f;
	CUDA_CHECK(cudaEventElapsedTime(&gemmMs, start, stop));
	gemmMs /= timedIters;

	float maxDiff = 0.f;
	for (int i = 0; i < m * p; i++)
		maxDiff = fmaxf(maxDiff, fabsf(naiveC[i] - gemmC[i]));

	std::cout << "m=n=p: " << m << "\n";
	std::cout << "matrix_product_GPU: " << naiveMs << " ms, "
		<< (flops / 1e9) / (naiveMs / 1e3) << " GFLOP/s\n";
	std::cout << "gemmGPU:            " << gemmMs << " ms, "
		<< (flops / 1e9) / (gemmMs / 1e3) << " GFLOP/s\n";
	std::cout << "speedup: " << naiveMs / gemmMs << "x\n";
	std::cout << "max abs diff between the two outputs: " << maxDiff << "\n";

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}
