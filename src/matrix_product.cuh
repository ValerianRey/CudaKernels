#pragma once

// Computes the matrix product out = A @ B, where A is [m, n] row-major,
// B is [n, p] row-major, and out is [m, p] row-major
// (out[i, j] = sum_k A[i, k] * B[k, j]).
// A, B and out must be device-accessible pointers (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least m*n, n*p and m*p floats respectively.
void matrix_product_GPU(int m, int n, int p, float* A, float* B, float* out);