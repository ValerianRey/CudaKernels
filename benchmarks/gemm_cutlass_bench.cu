#include "gemm_cutlass.cuh"
#include "gemm.cuh"
#include "matrix_product.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace {

float timeMs(cudaEvent_t start, cudaEvent_t stop, int warmupIters, int timedIters,
	void (*launch)(int, int, int, float *, float *, float *), int m, int n, int p,
	float *A, float *B, float *C)
{
	for (int i = 0; i < warmupIters; i++)
		launch(m, n, p, A, B, C);

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < timedIters; i++)
		launch(m, n, p, A, B, C);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	return ms / timedIters;
}

} // namespace

int main(void)
{
	int m = 2048, n = 2048, p = 2048;
	int warmupIters = 3;
	int timedIters = 10;

	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> naiveC(m * p);
	CudaManagedArray<float> tiledC(m * p);
	CudaManagedArray<float> cutlassC(m * p);

	for (int i = 0; i < m * n; i++)
		A[i] = static_cast<float>((i % 7) - 3) * 0.1f;
	for (int i = 0; i < n * p; i++)
		B[i] = static_cast<float>((i % 5) - 2) * 0.1f;

	A.prefetchToDevice(0, 0);
	B.prefetchToDevice(0, 0);
	naiveC.prefetchToDevice(0, 0);
	tiledC.prefetchToDevice(0, 0);
	cutlassC.prefetchToDevice(0, 0);
	CUDA_CHECK(cudaDeviceSynchronize());

	double flops = 2.0 * m * n * p; // one multiply + one add per inner-product term

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	float naiveMs = timeMs(start, stop, warmupIters, timedIters, matrix_product_GPU, m, n, p,
		A.get(), B.get(), naiveC.get());

	auto gemmWrapper = [](int m, int n, int p, float *A, float *B, float *C) {
		gemmGPU(m, n, p, 1.f, A, B, 0.f, C);
	};
	float tiledMs = timeMs(start, stop, warmupIters, timedIters, gemmWrapper, m, n, p, A.get(),
		B.get(), tiledC.get());

	auto cutlassWrapper = [](int m, int n, int p, float *A, float *B, float *C) {
		gemmCutlassGPU(m, n, p, 1.f, A, B, 0.f, C);
	};
	float cutlassMs = timeMs(start, stop, warmupIters, timedIters, cutlassWrapper, m, n, p,
		A.get(), B.get(), cutlassC.get());

	float tiledVsNaiveDiff = 0.f, cutlassVsTiledDiff = 0.f;
	for (int i = 0; i < m * p; i++) {
		tiledVsNaiveDiff = fmaxf(tiledVsNaiveDiff, fabsf(naiveC[i] - tiledC[i]));
		cutlassVsTiledDiff = fmaxf(cutlassVsTiledDiff, fabsf(tiledC[i] - cutlassC[i]));
	}

	std::cout << "m=n=p: " << m << "\n\n";
	std::cout << "matrix_product_GPU (naive):    " << naiveMs << " ms, "
		<< (flops / 1e9) / (naiveMs / 1e3) << " GFLOP/s\n";
	std::cout << "gemmGPU (hand-tiled):          " << tiledMs << " ms, "
		<< (flops / 1e9) / (tiledMs / 1e3) << " GFLOP/s\n";
	std::cout << "gemmCutlassGPU (tile-based):   " << cutlassMs << " ms, "
		<< (flops / 1e9) / (cutlassMs / 1e3) << " GFLOP/s\n\n";
	std::cout << "hand-tiled speedup over naive: " << naiveMs / tiledMs << "x\n";
	std::cout << "CUTLASS speedup over hand-tiled: " << tiledMs / cutlassMs << "x\n";
	std::cout << "CUTLASS speedup over naive: " << naiveMs / cutlassMs << "x\n\n";
	std::cout << "max abs diff, naive vs hand-tiled: " << tiledVsNaiveDiff << "\n";
	std::cout << "max abs diff, hand-tiled vs CUTLASS: " << cutlassVsTiledDiff << "\n";

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}
