#include <iostream>  
#include"hip/hip_runtime.h"
#include "hipblas.h"
using namespace std; 
int const M = 50;
int const N = 10;

int main()
{
	hipblasStatus_t status;
	hipblasHandle_t handle;
	status = hipblasCreate(&handle);
	if (status != HIPBLAS_STATUS_SUCCESS)
	{
		if (status == HIPBLAS_STATUS_NOT_INITIALIZED) {
			cout << "HIPBLAS 对象实例化出错" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}
    float** Matrix1;
    float* Matrix11;
    float** Matrix2;
    float* Matrix22;
    float** cpuresultMatrix;
    float* cpuresultMatrix1;
    float** gpuMatrix1;
    float* gpuMatrix11;
    float** gpuMatrix2;
    float* gpuMatrix22;
    float** gpuresultMatrix;
    float* gpuresultMatrix1;
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    Matrix1 = (float**)malloc(M* sizeof(float*));
    Matrix11 = (float*)malloc(M*N* sizeof(float));
    Matrix2 = (float**)malloc(N* sizeof(float*));
    Matrix22 = (float*)malloc(N*M* sizeof(float));
    cpuresultMatrix = (float**)malloc(M* sizeof(float*));
    cpuresultMatrix1 = (float*)malloc(M*M * sizeof(float));
    for (int i = 0; i < M; i++)
    {
      for(int j = 0; j < N; j++)
      {
        Matrix1[i][j] = (float)(rand() % 10 + 1);
        Matrix2[j][i] = (float)(rand() % 10 + 1);
      }
    }
    hipMalloc((void**)&gpuMatrix1, M* sizeof(float*));
    hipMalloc((void**)&gpuMatrix2, N* sizeof(float*));
    hipMalloc((void**)&gpuresultMatrix, M* sizeof(float*));
    hipMalloc((void**)&gpuMatrix11, M*N* sizeof(float));
    hipMalloc((void**)&gpuMatrix22, N*M* sizeof(float));
    hipMalloc((void**)&gpuresultMatrix1, M*M* sizeof(float));
    hipMemcpy(gpuMatrix1, Matrix1, M * sizeof(float*), hipMemcpyHostToDevice);    
    hipMemcpy(gpuMatrix2, Matrix2, N * sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(gpuMatrix11, Matrix11, M*N * sizeof(float), hipMemcpyHostToDevice);    
    hipMemcpy(gpuMatrix22, Matrix22, N*M * sizeof(float), hipMemcpyHostToDevice);
    hipDeviceSynchronize();
	  float a = 1; float b = 0;float batchCount=5;
	  hipblasSgemmBatched(
		  handle,
      HIPBLAS_OP_N, 
      HIPBLAS_OP_N,
      M,   
		  M,    
		  N,     
		  &a,    
		  gpuMatrix1,     
		  N,    
		  gpuMatrix2,    
		  M,    
		  &b,   
		  gpuresultMatrix,   
		  M,   
      batchCount)
	hipDeviceSynchronize();
 	hipMemcpy(gpuresultMatrix,cpuresultMatrix, M *sizeof(float*), hipMemcpyDeviceToHost);
  hipMemcpy(gpuresultMatrix1,cpuresultMatrix1, M*M *sizeof(float), hipMemcpyDeviceToHost);
    for (int i = 0; i < M; i++)
    {
      for(int j = 0;j<M;j++)
        cout << cpuresultMatrix[i][j] << " ";
      cout << endl;
    }
    hipFree(gpuMatrix11);
    hipFree(gpuMatrix22);
    hipFree(gpuresultMatrix1);
    hipFree(gpuMatrix1);
    hipFree(gpuMatrix2);
    hipFree(gpuresultMatrix);
    free(Matrix11);
    free(Matrix22);
    free(cpuresultMatrix1);
    free(Matrix1);
    free(Matrix2);
    free(cpuresultMatrix);
    return 0;
  }
