#pragma once

// Computes sum[i] = x[i] + y[i] for i in [0, N) on the GPU.
// x, y and sum must be device-accessible pointers (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least N floats each.
void addVectorsGPU(int N, float *x, float *y, float *sum);
