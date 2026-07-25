#pragma once

// Computes sum(x[i] * y[i]) for i in [0, N) on the GPU.
// x and y must be device-accessible pointers (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least N floats each.
float dotProductGPU(int N, float *x, float *y);
