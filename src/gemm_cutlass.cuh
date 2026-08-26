#pragma once

// Computes the general matrix product (GEMM) C = alpha * A @ B + beta * C,
// where A is [m, n] row-major, B is [n, p] row-major, and C is [m, p]
// row-major. Same contract as gemm.cu's gemmGPU (n == 0 only scales C by
// beta; beta == 0 never reads C, so it may hold NaN/uninitialized data),
// but built on NVIDIA CUTLASS's device::Gemm instead of a hand-written
// kernel: you describe a hierarchy of tile shapes (threadblock/warp/
// instruction) and CUTLASS generates and schedules the per-thread code,
// rather than you writing thread indices and shared-memory loads yourself.
// Uses CUTLASS's SIMT (plain FMA) operator class, since this project
// targets Pascal-class GPUs, which have no tensor cores.
// A, B and C must be device-accessible pointers (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least m*n, n*p and m*p floats respectively.
void gemmCutlassGPU(int m, int n, int p, float alpha, const float *A, const float *B, float beta, float *C);
