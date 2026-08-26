#include "matrix_product.cuh"

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#define BLOCK_DIM 16

__global__ void matrixProduct(int m, int n, int p, float *A, float *B, float *out)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m) return;
    if (col >= p) return;

    float sum = 0.f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * p + col];
    out[row * p + col] = sum;
}

void matrix_product_GPU(int m, int n, int p, float* A, float* B, float* out)
{
	if (m <= 0 || n <= 0 || p <= 0)
		return;

	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((p + BLOCK_DIM - 1) / BLOCK_DIM, (m + BLOCK_DIM - 1) / BLOCK_DIM);

	matrixProduct<<<gridDim, blockDim>>>(m, n, p, A, B, out);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}