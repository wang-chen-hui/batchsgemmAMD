#include <iostream>
#include "hip/hip_runtime.h"
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1
__global__ void matrixTranspose(float* out, float* in, const int width) {
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
	out[y * width + x] = in[x * width + y];
}
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
	for (unsigned int j = 0; j < width; j++) {
		for (unsigned int i = 0; i < width; i++) {
			output[i * width + j] = input[j * width + i];
		}
	}
}
int main() {
float* Matrix;
float* TransposeMatrix;
float* cpuTransposeMatrix;
float* gpuMatrix;
float* gpuTransposeMatrix;
hipDeviceProp_t devProp;
hipGetDeviceProperties(&devProp, 0);
std::cout << "Device name " << devProp.name << std::endl;
int i;
int errors;
Matrix = (float*)malloc(NUM * sizeof(float));
TransposeMatrix = (float*)malloc(NUM * sizeof(float));
cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));
for (i = 0; i < NUM; i++) {
	Matrix[i] = (float)i * 10.0f;
}
hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);
hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,gpuMatrix, WIDTH);
hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);
errors = 0;
double eps = 1.0E-6;
for (i = 0; i < NUM; i++) {
	if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
		errors++;
	}
}
if (errors != 0){
	printf("FAILED: %d errors\n", errors);
}
else{
	printf("PASSED!\n");
}
	hipFree(gpuMatrix);
	hipFree(gpuTransposeMatrix);
	free(Matrix);
	free(TransposeMatrix);
	free(cpuTransposeMatrix);
	return errors;
}
