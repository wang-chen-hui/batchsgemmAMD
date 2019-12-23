
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

__global__ void sgemm_strided_batched(sgemm_operation trans_a,
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
                                int i = threadIdx.x + blockIdx.x * blockDim.x;
                                int j = threadIdx.y + blockIdx.y * blockDim.y;
                                if (i < m && j < n)
                                {
                                    float accumulator=0;
                                    for (int K = 0; K < k; ++K)
                                    {
                                        accumulator +=A[j + i * lda] * B[K + j * ldb];
                                    }
                                    C[i + j * ldc] = (*alpha) * accumulator +C[i + j * ldc] * (*beta);
                                }
                           };
#endif
