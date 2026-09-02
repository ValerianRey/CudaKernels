#include "softmax_2d.cuh"

#define BLOCK_SIZE_X ((32))
#define BLOCK_SIZE_Y ((16))

__global__ void sum_exp(float* input, int rows, int cols, float* out) {
    __shared__ float shared_exp[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;

    if (by + y < rows && bx + x < cols) {
        shared_exp[y][x] = __expf(input[(by + y) * cols + bx + x]);
    } else {
        shared_exp[y][x] = 0.F;
    }

    for (int offset = BLOCK_SIZE_X / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (x < offset) {
            shared_exp[y][x] += shared_exp[y][x + offset];
        }
    }
    __syncthreads();
    if (x == 0) {
        atomicAdd(&out[by + y], shared_exp[y][0]);
    }
}


__global__ void exp_divide_per_row(float* input, int rows, int cols, float* denom) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;

    if (by + y >= rows || bx + x >= cols) {
        return;
    } else {
        int idx = (by + y) * cols + bx + x;
        input[idx] = __expf(input[idx]) / denom[by + y];
    }
}


void softmax_2d(float* input, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    float* denom = nullptr;
    cudaMallocManaged(&denom, rows * sizeof(float));
    for (int i = 0; i < rows; i++) {
        denom[i] = 0;
    }

    int num_blocks_x = (cols + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int num_blocks_y = (rows + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    dim3 grid_dim(num_blocks_x, num_blocks_y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    sum_exp<<<grid_dim, block_dim>>>(input, rows, cols, denom);
    cudaGetLastError();
    cudaDeviceSynchronize();

    exp_divide_per_row<<<grid_dim, block_dim>>>(input, rows, cols, denom);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaFree(denom);
}