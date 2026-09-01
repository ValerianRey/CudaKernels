#pragma once

#include <cuda_runtime.h>

#define BLOCK_SIZE ((32))  // This doesn't work with non-square blocks


template <typename T>
__global__ void transpose_kernel(T* input, T* output, int rows, int cols) {
    __shared__ T shared_arr[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    const int input_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int input_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_col = blockIdx.y * blockDim.y + threadIdx.x;
    const int output_row = blockIdx.x * blockDim.x + threadIdx.y;

    if (input_col < cols && input_row < rows) {
        shared_arr[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }
    
    __syncthreads();

    if (output_col < rows && output_row < cols) {
        output[output_row * rows + output_col] = shared_arr[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
void transpose(T* input, T* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    int num_blocks_x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_y = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid_dim(num_blocks_x, num_blocks_y);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

    transpose_kernel<<<grid_dim, block_dim, 0, 0>>>(input, output, rows, cols);
    cudaGetLastError();
    cudaStreamSynchronize(0);
}