#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void helloWorldFromGPU()
{
	printf("Hello world from GPU\n");
}

int main()
{
	cout<<"Hello world from CPU\n";
	helloWorldFromGPU<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
