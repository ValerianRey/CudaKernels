// Based on the excellent tutorial https://developer.nvidia.com/blog/even-easier-introduction-cuda/

#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

__global__ void add(int N, float *sum, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < N; i+=stride)
		sum[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1<<20;
	int blockSize, numBlocks;
	
	blockSize = 256;
	numBlocks = (N + blockSize - 1) / blockSize;

	// prompt user for block size and number of blocks
	//std::cout<<"Enter the desired block size (threads per block)"<<std::endl;
	//std::cin>>blockSize;
	
	//std::cout<<"Enter the desired number of blocks"<<std::endl;
	//std::cin>>numBlocks;

	std::cout<<"Computing vector sum with "<<numBlocks<<" blocks of size "<<blockSize<<std::endl;

	float *x, *y, *sum;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&sum, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(sum, N*sizeof(float), 0, 0);

	add<<<numBlocks, blockSize>>>(N, sum, x, y);
	
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(sum[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);
	cudaFree(sum);

	return 0;
}
