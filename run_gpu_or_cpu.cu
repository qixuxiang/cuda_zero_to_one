#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
 

__host__ __device__ int run_on_cpu_or_gpu() {
	return 1;
}
 
__global__ void run_on_gpu() {
	printf("run_on_cpu_or_gpu GPU: %d\n", run_on_cpu_or_gpu());
}
 
int main() {

	printf("run_on_cpu_or_gpu CPU: %d\n", run_on_cpu_or_gpu());
	run_on_gpu<<<1, 1>>>();
	printf("will end\n");
	cudaDeviceReset();
	return 0;
}