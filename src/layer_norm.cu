#include "layer_norm.cuh"


#define BLOCK_SIZE_X  ((8))
#define BLOCK_SIZE_Y  ((8))
#define BLOCK_SIZE_Z  ((8))


__global__ void compute_stats(float* input, int batch, int rows, int cols, float* sums, float* sq_sums) {
    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int stat_idx = b * rows + r;
    const int idx = b * rows * cols + r * cols + c;

    __shared__ float shared_sums[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float shared_sq_sums[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];

    if (b < batch && r < rows && c < cols) {
        float inp = input[idx];  // coalesced read
        shared_sums[threadIdx.z][threadIdx.y][threadIdx.x] = inp;  
        shared_sq_sums[threadIdx.z][threadIdx.y][threadIdx.x] = inp * inp;
    } else {
        shared_sums[threadIdx.z][threadIdx.y][threadIdx.x] = 0.F;
        shared_sq_sums[threadIdx.z][threadIdx.y][threadIdx.x] = 0.F;
    }

    for (int offset = BLOCK_SIZE_X / 2; offset > 0; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shared_sums[threadIdx.z][threadIdx.y][threadIdx.x] += shared_sums[threadIdx.z][threadIdx.y][threadIdx.x + offset];
            shared_sq_sums[threadIdx.z][threadIdx.y][threadIdx.x] += shared_sq_sums[threadIdx.z][threadIdx.y][threadIdx.x + offset];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && b < batch && r < rows) {
        atomicAdd(&sums[stat_idx], shared_sums[threadIdx.z][threadIdx.y][0]);
        atomicAdd(&sq_sums[stat_idx], shared_sq_sums[threadIdx.z][threadIdx.y][0]);
    }
}


__global__ void layer_norm_kernel_ew(float* input, int batch, int rows, int cols, float* weight, float* bias, float* sums, float* sq_sums, float eps) {
    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || r >= rows || c >= cols) return;

    const int stat_idx = b * rows + r;
    const int idx = b * rows * cols + r * cols + c;

    float mean = sums[stat_idx] / cols;
    float var = sq_sums[stat_idx] / cols - mean * mean;
    input[idx] = bias[c] + weight[c] * (input[idx] - mean) / sqrt(var + eps);
}


// In-place, affine layer normalization of a [batch, rows, cols] row-major
// tensor, matching PyTorch's nn.LayerNorm with normalized_shape = (cols,).
// For each (b, r) pair, normalizes across the last dimension (cols) and then
// applies a per-column affine transform:
// out[b, r, c] = (input[b, r, c] - mean) / sqrt(var + eps) * weight[c] + bias[c],
// for c in [0, cols), where mean and var (biased, i.e. divided by cols) are
// computed over that same range.
// input must be a device-accessible pointer (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least batch*rows*cols floats.
// weight and bias must be device-accessible pointers holding at least cols
// floats each.
void layer_norm(float* input, int batch, int rows, int cols, float* weight, float* bias, float eps) {
    if (cols <= 0 || rows <= 0 || batch <= 0) return;

    const int num_blocks_x = (cols + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    const int num_blocks_y = (rows + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    const int num_blocks_z = (batch + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size(num_blocks_x, num_blocks_y, num_blocks_z);

    float* sums = nullptr;
    float* sq_sums = nullptr;
    cudaMallocManaged(&sums, sizeof(float) * batch * rows);
    cudaMallocManaged(&sq_sums, sizeof(float) * batch * rows);
    for (int i = 0; i < batch * rows; i++) {
        sums[i] = 0.F;
        sq_sums[i] = 0.F;
    }

    compute_stats<<<grid_size, block_size>>>(input, batch, rows, cols, sums, sq_sums);
    cudaGetLastError();
    cudaDeviceSynchronize();

    layer_norm_kernel_ew<<<grid_size, block_size>>>(input, batch, rows, cols, weight, bias, sums, sq_sums, eps);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaFree(sums);
    cudaFree(sq_sums);
}
