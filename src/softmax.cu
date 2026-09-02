#include "softmax.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 256


__global__ void sum_exp(int N, float* input, float* out) {
    int x = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;

    __shared__ float shared_exp[BLOCK_SIZE];

    if (bx + x < N) {
        shared_exp[x] = __expf(input[bx + x]);
    } else {
        shared_exp[x] = 0.F;
    }

    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset /= 2) {
        __syncthreads();
        if (x < offset) {
            shared_exp[x] += shared_exp[x + offset];
        }
    }
    __syncthreads();

    if (x < 32) {
        float sum_result = shared_exp[x];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_result += __shfl_down_sync(0xffffffff, sum_result, offset);
        }
        if (x == 0) {
            atomicAdd(out, sum_result);
        }
    }
}

__global__ void divide_pointwise(int N, float* input, float denom) {
    int x = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;

    if (bx + x >= N) {
        return;
    }

    input[bx + x] = __expf(input[bx + x]) / denom;
}


void softmax_(int N, float* input) {
    if (N <= 0) {
        return;
    }

    float* denom = nullptr;
    cudaMallocManaged(&denom, sizeof(float));
    *denom = 0.F;

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    sum_exp<<<num_blocks, BLOCK_SIZE, 0, 0>>>(N, input, denom);

    cudaGetLastError();
    cudaStreamSynchronize(0);

    divide_pointwise<<<num_blocks, BLOCK_SIZE, 0, 0>>>(N, input, *denom);

    cudaGetLastError();
    cudaStreamSynchronize(0);

    cudaFree(denom);
}