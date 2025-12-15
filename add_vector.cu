#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

__global__ void add(int N, float *sum, float *x, float *y)
{
	for (int i = 0; i < N; i++)
		sum[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1<<20;

	float *x, *y, *sum;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&sum, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add<<<1, 1>>>(N, sum, x, y);
	
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
