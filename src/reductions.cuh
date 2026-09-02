#pragma once

#include "managed_array.cuh"
#include "cuda_check.cuh"


#define BLOCK_SIZE ((256))
#define ELEMS_PER_BLOCK  ((BLOCK_SIZE) * 2)

template <typename T>
__global__ void sum_reduction_kernel(int N, T* input, T* out) {
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
T sum_reduction(int N, T* input) {
    T* out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(T)));
    *out = (T) 0;
    int num_blocks = N > 0 ? (N + BLOCK_SIZE - 1) / BLOCK_SIZE : 1;
    sum_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(N, input, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T result = *out;
    CUDA_CHECK(cudaFree(out));
    return result;
}


template <typename T>
__global__ void max_reduction_kernel(int N, T* input, T* out) {
    int x = threadIdx.x;
    int b = blockIdx.x * ELEMS_PER_BLOCK;
    
    __shared__ T buffer[BLOCK_SIZE];

    if (x + b + BLOCK_SIZE < N) {
        buffer[x] = max(input[x + b], input[x + b + BLOCK_SIZE]);
    } else if (x + b < N) {
        buffer[x] = input[x + b];
    }
    else {
        buffer[x] = *out;
    }

    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (x < offset) {
            buffer[x] = max(buffer[x], buffer[x + offset]);
        }
    }
    __syncwarp();

    if (x < 32) {
        T result = buffer[x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            result = max(result, __shfl_down_sync(0xffffffff, result, offset));
        }
        if (x == 0) {
            atomicMax(out, result);
        }
    }
}



template <typename T>
T max_reduction(int N, T* input) {
    T* out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(T)));
    *out = std::numeric_limits<T>::lowest();

    int num_blocks = N == 0? 1 : (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

    max_reduction_kernel<<<num_blocks, BLOCK_SIZE, /*dynamic shared mem per block=*/0, /*stream=*/0>>>(N, input, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));

    T result = *out;
    CUDA_CHECK(cudaFree(out));
    return result;
}