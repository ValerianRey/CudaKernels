#include "matrix_product.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <iostream>
#include <cuda_runtime.h>

int main(void)
{
	int m = 10;
	int n = 20;
	int p = 8;

	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> C(m * p);

	for (int i = 0; i < m * n; i++) {
		A[i] = 3.0f;
	}

	for (int i = 0; i < n * p; i++) {
		B[i] = 5.0f;
	}

	// Prefetch to the GPU
	A.prefetchToDevice(0, 0);
	B.prefetchToDevice(0, 0);
	C.prefetchToDevice(0, 0);

	matrix_product_GPU(m, n, p, A.get(), B.get(), C.get());

	float expectedResult = n * 15.0f;

	for (int i = 0; i < m * p; i++) {
		float error = expectedResult - C[i];
		std::cout<<"Error: "<<error<<std::endl;
	}

	return 0;
}
