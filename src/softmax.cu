#include "softmax.cuh"
#include "cuda_check.cuh"
#include "managed_array.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 256


__global__ void sum_exp_reduction_tree(int N, float* input, float* buf) {
    int x = threadIdx.x;
    int b = blockIdx.x * blockDim.x;
    
    __shared__ float block_buf[BLOCK_SIZE];
    if (x + b < N) {
        block_buf[x] = exp(input[b + x]);
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
        buf[blockIdx.x] = block_buf[0];
    }
}

__global__ void inplace_softmax(int N, float* input, float denom_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    input[i] = exp(input[i]) * denom_inv;
}

void softmax_(int N, float* input) {
    if (N <= 0) {
        return;
    }

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CudaManagedArray<float> buf(num_blocks);

    sum_exp_reduction_tree<<<num_blocks, BLOCK_SIZE>>>(N, input, buf.get());

    CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

    float buf_sum = 0.F;
    for (int i = 0; i < num_blocks; i++) {
        buf_sum += buf[i];
    }

    float denom_inv = 1.F / buf_sum;

    inplace_softmax<<<num_blocks, BLOCK_SIZE>>>(N, input, denom_inv);

    CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}