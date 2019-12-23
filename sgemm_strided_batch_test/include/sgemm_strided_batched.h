
/**/
#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include<iostream>
#include "hip/hip_runtime.h"
#include <time.h> 
using namespace std;
typedef enum sgemm_operation_ {
    operation_none      = 0, 
    operation_transpose  = 1,
    operation_conjugate_transpose = 2
} sgemm_operation;

__global__ void ReferenceGemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta,
    float *C,
    int ldc)
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
                                C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
                            }
                           }
void sgemm_strided_batched(sgemm_operation trans_a,
                           sgemm_operation trans_b,
                           int m,
                           int n,
                           int k,
                           const float* alpha,
                           const float* A,
                           int lda,
                           int stride_a,
                           const float* B,
                           int ldb,
                           int stride_b,
                           const float* beta,
                           float* C,
                           int ldc,
                           int stride_c,
                           int batch_count)
{
    //cout << "compute sgemm stride batch" << endl;
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    /*if (trans_a == operation_none && trans_b == operation_none)
    {
        cout << "NO transpose" << endl;
    }
    else
    {
        cout << "Transpose!" << endl;
    }
    */
    for (int i = 0; i < batch_count; i++)
    {
        hipLaunchKernelGGL(ReferenceGemm_kernel, grid, block, 0 , 0 ,
            m,
            n,
            k,
            *alpha,
            A,
            lda,
            B,
            ldb,
            *beta,
            C,
            ldc);
        A += stride_a;
        B += stride_b;
        C += stride_c;
    }
}

#endif
