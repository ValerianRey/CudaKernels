#pragma once

#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE ((256))


template<typename T>
__global__ void min_max_reduction(int N, T* input, T* out_min, T* out_max, T min_T, T max_T) {
    __shared__ T block_mins[BLOCK_SIZE];
    __shared__ T block_maxes[BLOCK_SIZE];

    int x = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;
    if (bx + x >= N) {
        block_mins[x] = max_T;
        block_maxes[x] = min_T;
    } else {
        T val = input[bx + x];
        block_mins[x] = val;
        block_maxes[x] = val;
    }

    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset /= 2) {
        __syncthreads();
        if (x < offset) {
            block_mins[x] = min(block_mins[x], block_mins[x + offset]);
            block_maxes[x] = max(block_maxes[x], block_maxes[x + offset]);
        }
    }
    __syncthreads();

    if (x < 32) {
        T result_min = block_mins[x];
        T result_max = block_maxes[x];
        for (int offset = 16; offset > 0; offset /= 2) {
            result_min = min(result_min, __shfl_down_sync(0xffffffff, result_min, offset));
            result_max = max(result_max, __shfl_down_sync(0xffffffff, result_max, offset));
        }
        if (x == 0) {
            atomicMin(out_min, result_min);
            atomicMax(out_max, result_max);
        }
    }
}

template<typename T>
__global__ void histogram_kernel(int N, T* input, int* result, int n_bins, T out_min, T bin_size) {
    extern __shared__ int block_result[];
    int x = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;
    int bin_id;
    for (int i = 0; i < (n_bins + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        bin_id = i * BLOCK_SIZE + x;
        if (bin_id < n_bins) {
            block_result[bin_id] = 0;
        }
    }
    __syncthreads();
    
    if (bx + x < N) {
        T num = input[bx + x];
        bin_id = (num - out_min) / bin_size;
        if (bin_id >= n_bins) {
            bin_id = n_bins - 1;
        }
        atomicAdd(&block_result[bin_id], 1);
    }

    __syncthreads();

    for (int i = 0; i < (n_bins + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        bin_id = i * BLOCK_SIZE + x;
        if (bin_id < n_bins) {
            atomicAdd(&result[bin_id], block_result[bin_id]);
        }
    }
}


template<typename T>
void histogram(int N, T* input, int n_bins, int* result) {
    T* out_min = nullptr;
    T* out_max = nullptr;
    cudaMallocManaged(&out_min, sizeof(T));
    cudaMallocManaged(&out_max, sizeof(T));
    for (int i = 0; i < n_bins; i++) {
        result[i] = 0;
    }
    *out_min = std::numeric_limits<T>::max();
    *out_max = std::numeric_limits<T>::lowest();

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    min_max_reduction<<<num_blocks, BLOCK_SIZE, 0, 0>>>(N, input, out_min, out_max, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
    cudaGetLastError();
    cudaStreamSynchronize(0);

    T bin_size = (*out_max - *out_min + n_bins - 1) / n_bins;
    if (bin_size == (T) 0) {
        bin_size = (T) 1;
    }

    histogram_kernel<<<num_blocks, BLOCK_SIZE, n_bins * sizeof(int), 0>>>(N, input, result, n_bins, *out_min, bin_size);
    cudaGetLastError();
    cudaStreamSynchronize(0);

    cudaFree(out_min);
    cudaFree(out_max);
}