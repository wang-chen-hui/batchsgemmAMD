
#include "include/sgemm_strided_batched.h"
//#include "hip/hip_runtime.h"
#include "cuda_runtime.h"
using namespace std;

__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

void sgemm_strided_batched(sgemm_operation trans_a,
                           sgemm_operation trans_b,
                           int m,
                           int n,
                           int k,
                           const float *alpha,
                           const float *A,
                           int lda,
                           int stride_a,
                           const float *B,
                           int ldb,
                           int stride_b,
                           const float *beta,
                           float *C,
                           int ldc,
                           int stride_c,
                           int batch_count)
{
    //cout << "compute sgemm stride batch" << endl;
    
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    if(trans_a == operation_none && trans_b == operation_none) 
    {
        cout << "NO transpose" << endl;
    }
    else
    {
        cout << "Transpose!" << endl;
    }


    for (int i = 0; i < batch_count; i++)
    {
        A += stride_a;
        B += stride_b;
        C += stride_c;
        cudaLaunchKernelGGL(ReferenceGemm_kernel,
                           grid,
                           block,
                           0,
                           0,
                           m,
                           n,
                           k,
                           alpha,
                           A,
                           lda,
                           B,
                           ldb,
                           C,
                           ldc);
    }

    //cout << "compute sgemm stride batch" << endl;
}
