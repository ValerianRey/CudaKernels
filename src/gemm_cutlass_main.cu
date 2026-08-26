#include "gemm_cutlass.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main(void)
{
	int m = 10;
	int n = 20;
	int p = 8;
	float alpha = 2.0f;
	float beta = 10.0f;

	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> C(m * p);

	for (int i = 0; i < m * n; i++) {
		A[i] = 3.0f;
	}

	for (int i = 0; i < n * p; i++) {
		B[i] = 5.0f;
	}

	for (int i = 0; i < m * p; i++) {
		C[i] = 1.0f;
	}

	// Prefetch to the GPU
	A.prefetchToDevice(0, 0);
	B.prefetchToDevice(0, 0);
	C.prefetchToDevice(0, 0);

	gemmCutlassGPU(m, n, p, alpha, A.get(), B.get(), beta, C.get());

	// C = alpha * (A @ B) + beta * C, with every A@B entry equal to n * 3 * 5.
	float expectedResult = alpha * (n * 15.0f) + beta * 1.0f;

	for (int i = 0; i < m * p; i++) {
		float error = expectedResult - C[i];
		std::cout << "Error: " << error << std::endl;
	}

	return 0;
}
