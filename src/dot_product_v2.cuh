#define BLOCK_SIZE 256


template <typename T>
__device__ void sum_reduction(T* shared_out_block, T* out) {
    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shared_out_block[threadIdx.x] += shared_out_block[threadIdx.x + offset];
        }
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        T result = shared_out_block[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xffffffff, result, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(out, result);
        }
    }
}

template <typename T>
__global__ void dot_product_kernel(int N, T* a, T* b, T* out) {
    __shared__ T out_block[BLOCK_SIZE];

    int x = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;
    
    if (x + bx < N) {
        out_block[x] = a[bx + x] * b[bx + x];
    } else {
        out_block[x] = (T) 0;
    }

    sum_reduction(out_block, out);
}


template <typename T>
T dot_product(int N, T* a, T* b) {
    if (N == 0) {
        return (T) 0;
    }

    T* out = nullptr;
    cudaMallocManaged(&out, sizeof(T));
    *out = (T) 0;

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dot_product_kernel<<<num_blocks, BLOCK_SIZE, 0, 0>>>(N, a, b, out);
    cudaGetLastError();
    cudaStreamSynchronize(0);

    T result = *out;
    cudaFree(out);
    return result;
}