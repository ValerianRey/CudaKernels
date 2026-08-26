#include "gemm.cuh"

#include "cuda_check.cuh"

#include <cuda_runtime.h>

// Classic shared-memory-tiled, register-blocked SGEMM: each block computes a
// BLOCK_M x BLOCK_P tile of C, sweeping the reduction dimension (n) in
// BLOCK_K-deep chunks staged through shared memory; each thread owns a
// THREAD_M x THREAD_P sub-tile of the block's output, accumulated in
// registers, so every value loaded into shared memory is reused THREAD_M *
// THREAD_P times instead of once (the naive kernel's global-memory traffic).
#define BLOCK_M 64
#define BLOCK_P 64
#define BLOCK_K 16
#define THREAD_M 4
#define THREAD_P 4
#define THREADS_PER_BLOCK ((BLOCK_M / THREAD_M) * (BLOCK_P / THREAD_P))

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
	gemm(int m, int n, int p, float alpha, const float *A, const float *B, float beta, float *C)
{
	__shared__ float As[BLOCK_M][BLOCK_K];
	__shared__ float Bs[BLOCK_K][BLOCK_P];

	int blockRow = blockIdx.y * BLOCK_M;
	int blockCol = blockIdx.x * BLOCK_P;

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int threadRow = threadIdx.y * THREAD_M;
	int threadCol = threadIdx.x * THREAD_P;

	// THREADS_PER_BLOCK threads don't cover a whole tile in one step (it has
	// more elements than there are threads), so each load sweeps the tile in
	// several strided passes.
	const int aLoadRowStride = THREADS_PER_BLOCK / BLOCK_K;
	const int bLoadRowStride = THREADS_PER_BLOCK / BLOCK_P;

	float acc[THREAD_M][THREAD_P] = {};

	for (int kStart = 0; kStart < n; kStart += BLOCK_K) {
		for (int r = tid / BLOCK_K; r < BLOCK_M; r += aLoadRowStride) {
			int c = tid % BLOCK_K;
			int globalRow = blockRow + r;
			int globalCol = kStart + c;
			As[r][c] = (globalRow < m && globalCol < n) ? A[globalRow * n + globalCol] : 0.f;
		}
		for (int r = tid / BLOCK_P; r < BLOCK_K; r += bLoadRowStride) {
			int c = tid % BLOCK_P;
			int globalRow = kStart + r;
			int globalCol = blockCol + c;
			Bs[r][c] = (globalRow < n && globalCol < p) ? B[globalRow * p + globalCol] : 0.f;
		}
		__syncthreads();

		#pragma unroll
		for (int kk = 0; kk < BLOCK_K; kk++) {
			float aFrag[THREAD_M];
			float bFrag[THREAD_P];

			#pragma unroll
			for (int i = 0; i < THREAD_M; i++)
				aFrag[i] = As[threadRow + i][kk];
			#pragma unroll
			for (int j = 0; j < THREAD_P; j++)
				bFrag[j] = Bs[kk][threadCol + j];

			#pragma unroll
			for (int i = 0; i < THREAD_M; i++)
				#pragma unroll
				for (int j = 0; j < THREAD_P; j++)
					acc[i][j] += aFrag[i] * bFrag[j];
		}
		__syncthreads();
	}

	#pragma unroll
	for (int i = 0; i < THREAD_M; i++) {
		int row = blockRow + threadRow + i;
		if (row >= m)
			continue;
		#pragma unroll
		for (int j = 0; j < THREAD_P; j++) {
			int col = blockCol + threadCol + j;
			if (col >= p)
				continue;
			int idx = row * p + col;
			// beta == 0 must not read C: it may be NaN or uninitialized,
			// per the standard BLAS gemm contract.
			C[idx] = (beta == 0.f) ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[idx];
		}
	}
}

void gemmGPU(int m, int n, int p, float alpha, const float *A, const float *B, float beta, float *C)
{
	if (m <= 0 || p <= 0)
		return;

	dim3 blockDim(BLOCK_P / THREAD_P, BLOCK_M / THREAD_M);
	dim3 gridDim((p + BLOCK_P - 1) / BLOCK_P, (m + BLOCK_M - 1) / BLOCK_M);

	gemm<<<gridDim, blockDim>>>(m, n, p, alpha, A, B, beta, C);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}
