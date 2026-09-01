#include "softmax.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 256


__global__ void sum_exp_reduction_tree(int N, float* input, float* denom) {
    int x = threadIdx.x;
    int b = blockIdx.x * blockDim.x;
    
    __shared__ float block_buf[BLOCK_SIZE];
    if (x + b < N) {
        input[b + x] = exp(input[b + x]);
        block_buf[x] = input[b + x];
    } else {
        block_buf[x] = 0.F;
    }

    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (x < stride) {
            block_buf[x] += block_buf[x + stride];
        }
    }
    __syncthreads();
    if (x == 0) {
        atomicAdd(denom, block_buf[0]);
    }
}

__global__ void inplace_softmax(int N, float* input, const float* denom) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    input[i] = input[i] / *denom;
}

void softmax_(int N, float* input) {
    if (N <= 0) {
        return;
    }

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CudaManagedArray<float> denom(1);
    denom[0] = 0.F;
    sum_exp_reduction_tree<<<num_blocks, BLOCK_SIZE>>>(N, input, denom.get());

    CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

    inplace_softmax<<<num_blocks, BLOCK_SIZE>>>(N, input, denom.get());

    CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}