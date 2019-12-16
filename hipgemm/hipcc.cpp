#include<iostream>
#include <vector>
#include "hip/hip_runtime.h"
#include <time.h> 
using namespace std;
int const M = 50;
int const N = 10;
__global__ void ReferenceGemm_kernel(int M,int N,int K,float alpha,float const *A,int lda,float const *B,int ldb,float *C,int ldc) 
{
  int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int j = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
  if (i < N && j < M) 
  {
    float accumulator = 0;
    for (int k = 0; k < K; ++k)
    {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }
    C[i + j * ldc] = alpha * accumulator;
  }
}
int main() {
    float* Matrix1;
    float* Matrix2;
    float* cpuresultMatrix;
    float* gpuMatrix1;
    float* gpuMatrix2;
    float* gpuresultMatrix;
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    Matrix1 = (float*)malloc(M*N * sizeof(float));
    Matrix2 = (float*)malloc(N*M * sizeof(float));
    cpuresultMatrix = (float*)malloc(M*M * sizeof(float));
    for (int i = 0; i < N*M; i++)
    {
      Matrix1[i] = (float)(rand() % 10 + 1);
      Matrix2[i] = (float)(rand() % 10 + 1);
    }
    hipMalloc((void**)&gpuMatrix1, M*N * sizeof(float));
    hipMalloc((void**)&gpuMatrix2, N*M * sizeof(float));
    hipMalloc((void**)&gpuresultMatrix, M*M * sizeof(float));
    hipMemcpy(gpuMatrix1, Matrix1, M*N * sizeof(float), hipMemcpyHostToDevice);    
    hipMemcpy(gpuMatrix2, Matrix2, N*M * sizeof(float), hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    float num = 1;
    hipLaunchKernelGGL(ReferenceGemm_kernel, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0,M,   
    N,   
    M,     
    num,     
    gpuMatrix1,    
    N,    
    gpuMatrix2,     
    M,     
    gpuresultMatrix,     
    M,   );
    hipDeviceSynchronize();
    hipMemcpy(gpuresultMatrix,cpuresultMatrix, M*M * sizeof(float), hipMemcpyDeviceToHost);
    for (int i = 0; i < M*M; i++)
    {
      cout << cpuresultMatrix[i] << " ";
      if ((i + 1) % M == 0) cout << endl;
    }
    hipFree(gpuMatrix1);
    hipFree(gpuMatrix2);
    hipFree(gpuresultMatrix);
    free(Matrix1);
    free(Matrix2);
    free(cpuresultMatrix);
    return 0;
  }