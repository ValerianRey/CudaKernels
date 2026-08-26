#include "gemm_cutlass.cuh"

#include "cuda_check.cuh"

#include <cutlass/gemm/device/gemm.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUTLASS's tile hierarchy for this GEMM: OpClassSimt selects plain FMA
// (no tensor cores, since Pascal has none); Sm61 matches this project's
// GTX 1080. Threadblock/warp/instruction tile shapes are left at CUTLASS's
// own defaults for this (SIMT, fp32) configuration.
using CutlassGemm = cutlass::gemm::device::Gemm<
	float, cutlass::layout::RowMajor, // A
	float, cutlass::layout::RowMajor, // B
	float, cutlass::layout::RowMajor, // C
	float, // accumulator element type
	cutlass::arch::OpClassSimt,
	cutlass::arch::Sm61>;

void gemmCutlassGPU(int m, int n, int p, float alpha, const float *A, const float *B, float beta, float *C)
{
	if (m <= 0 || p <= 0)
		return;

	int lda = n; // row-major A is [m, n]: row stride is n
	int ldb = p; // row-major B is [n, p]: row stride is p
	int ldc = p; // row-major C is [m, p]: row stride is p

	CutlassGemm gemmOperator;
	CutlassGemm::Arguments args(
		{m, p, n}, // CUTLASS's problem_size is {M, N, K}: M=m, N=p, K=n (reduction)
		{A, lda},
		{B, ldb},
		{C, ldc},
		{C, ldc},
		{alpha, beta});

	cutlass::Status status = gemmOperator(args);
	if (status != cutlass::Status::kSuccess) {
		std::fprintf(stderr, "CUTLASS GEMM error at %s:%d: %s\n", __FILE__, __LINE__,
			cutlassGetStatusString(status));
		std::exit(EXIT_FAILURE);
	}
	CUDA_CHECK(cudaDeviceSynchronize());
}
