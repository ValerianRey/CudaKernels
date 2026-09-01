#pragma once

#include "managed_array.cuh"
#include "cuda_check.cuh"


#define BLOCK_SIZE ((256))

template <typename T>
__global__ void sum_kernel(int N, T* input, T* out) {
    int x = threadIdx.x;
    int b = blockIdx.x * blockDim.x;

    __shared__ T buffer[BLOCK_SIZE];
    if (x + b < N) {
        buffer[x] = input[b + x];
    } else {
        buffer[x] = (T) 0;
    }

    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (x < offset) {
            buffer[x] += buffer[x + offset];
        }
    }
    __syncthreads();

    if (x < 32) {
        // When we're done to reducing a single warp, we can use the faster warp-primitive instead
        T result = buffer[x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(0xffffffff, result, offset);
        }

        if (x == 0) {
            atomicAdd(out, result);
        }
    }
}


template <typename T>
T sum(int N, T* input) {
    T* out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(T)));
    *out = (T) 0;
    int num_blocks = N > 0 ? (N + BLOCK_SIZE - 1) / BLOCK_SIZE : 1;
    sum_kernel<<<num_blocks, BLOCK_SIZE>>>(N, input, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T result = *out;
    CUDA_CHECK(cudaFree(out));
    return result;
}