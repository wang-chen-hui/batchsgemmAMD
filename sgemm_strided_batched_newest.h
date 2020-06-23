#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include<iostream>
#include "hip/hip_runtime.h"
#include <time.h>

using namespace std;

#define BLOCK_SIZE1 4
#define BLOCK_SIZE2 16
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
    int stride_a,
    const float *B,
    int ldb,
    int stride_b,
    float beta,
    float *C,
    int ldc,
    int stride_c)
{
    int batch_id = hipBlockIdx_z;
    A += stride_a * batch_id;
    B += stride_b * batch_id;
    C += stride_c * batch_id;


    int row = hipThreadIdx_x + hipThreadIdx_y * BLOCK_SIZE1 + hipBlockIdx_x * (BLOCK_SIZE1 * BLOCK_SIZE2);
    int col = hipBlockIdx_y * BLOCK_SIZE2;

    __shared__ float Bs[BLOCK_SIZE1*2][BLOCK_SIZE2];
    float accumulator[BLOCK_SIZE2];

    for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx)
    {
        accumulator[outIdx] = 0;
    }


        int i = hipThreadIdx_y;
        int j = hipThreadIdx_x;
        if (j < K && col + i < N) 
        {
            Bs[j][i] = B[j + (col + i) * ldb];
        } 
        else {
            Bs[j][i] = 0;
        }
        __syncthreads();
    for (unsigned int tileIdx = 1; tileIdx < ((K - 1) / BLOCK_SIZE1 + 1);  tileIdx+=2)
    {
        if (tileIdx * BLOCK_SIZE1 + j < K && col + i < N) 
        {
            Bs[j+BLOCK_SIZE1][i] = B[tileIdx * BLOCK_SIZE1 + j + (col + i) * ldb];
        } 
        else
        {
            Bs[j+BLOCK_SIZE1][i] = 0;
        }   
        for (unsigned int idx = 0; idx < BLOCK_SIZE1; ++idx)
        {
            float a_reg;
	        if (row < M && (tileIdx-1) * BLOCK_SIZE1 + idx < K)
            {
                a_reg = A[row + ((tileIdx - 1) * BLOCK_SIZE1 + idx) * lda];
            }
	        else
	        {
		        a_reg = 0;
	        } 

            #pragma unroll
            for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx)
            {
                accumulator[outIdx] += a_reg * Bs[idx][outIdx];
            }
        }
        __syncthreads();
        if ((tileIdx + 1) * BLOCK_SIZE1 + j < K && col + i < N) 
        {
            Bs[j][i] = B[(tileIdx + 1) * BLOCK_SIZE1 + j + (col + i) * ldb];
        } 
        else
        {
            Bs[j][i] = 0;
        }   
        for (unsigned int idx = 0; idx < BLOCK_SIZE1; ++idx)
        {
            float a_reg;
	        if (row < M && tileIdx * BLOCK_SIZE1 + idx < K)
            {
                a_reg = A[row + (tileIdx * BLOCK_SIZE1 + idx) * lda];
            }
	        else
	        {
		        a_reg = 0;
	        } 

            #pragma unroll
            for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx)
            {
                accumulator[outIdx] += a_reg * Bs[idx + BLOCK_SIZE1][outIdx];
            }
        }
        __syncthreads();
    }

   #pragma unroll
   for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx) 
   {
        if (row < M && col + outIdx < N)
        {
            C[row + (col + outIdx) * ldc] = alpha * accumulator[outIdx] + beta * C[row + (col + outIdx) * ldc];
        }
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
    dim3 block(BLOCK_SIZE1, BLOCK_SIZE2);
    dim3 grid(
        (m + (block.x * block.y) - 1) / (block.x * block.y),
        (n + block.y - 1) / block.y,
        batch_count);

    hipLaunchKernelGGL(ReferenceGemm_kernel, grid, block, 0, 0,
                            m,
                            n,
                            k,
                            *alpha,
                            A,
                            lda,
                            stride_a,
                            B,
                            ldb,
                            stride_b,
                            *beta,
                            C,
                            ldc,
                            stride_c);
}
#endif
