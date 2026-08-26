#pragma once

// Computes the general matrix product (GEMM) C = alpha * A @ B + beta * C,
// where A is [m, n] row-major, B is [n, p] row-major, and C is [m, p]
// row-major (C[i, j] = alpha * sum_k A[i, k] * B[k, j] + beta * C[i, j]).
// A, B and C must be device-accessible pointers (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least m*n, n*p and m*p floats respectively.
// Unlike matrix_product_GPU, n == 0 is well-defined here (C is simply
// scaled by beta), matching the BLAS convention. beta == 0.f is also
// special-cased to never read C, so C may hold NaN/uninitialized data.
void gemmGPU(int m, int n, int p, float alpha, const float *A, const float *B, float beta, float *C);
