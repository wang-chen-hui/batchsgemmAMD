#include<stdio.h>

void CPUFunction()
{
	printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
	printf("This function is defined to run on the GPU.\n");
	printf("This function is defined to run on the GPU.\n");
	printf("This function is defined to run on the GPU.\n");
}

int main()
{
	CPUFunction();
	GPUFunction << <10, 120>> > ();
	cudaDeviceSynchronize();
}
