#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void run_on_gpu() {
	printf("GPU thread info X:%d Y:%d Z:%d\t block info X:%d Y:%d Z:%d\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}
 
int main() {
	dim3 threadsPerBlock(2, 3, 4);
    int blocksPerGrid = 1;
    
    run_on_gpu<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceReset();
	return 0;
}
