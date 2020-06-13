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
    //int rc = hipBlockIdx_x * (BLOCK_SIZE1 * BLOCK_SIZE2);

    __shared__ float Bs[BLOCK_SIZE1][BLOCK_SIZE2];
    //__shared__ float As[BLOCK_SIZE1][BLOCK_SIZE1*BLOCK_SIZE2];
    float accumulator[BLOCK_SIZE2];

    for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx)
    {
        accumulator[outIdx] = 0;
    }

    for (unsigned int tileIdx = 0; tileIdx < ((K - 1) / BLOCK_SIZE1 + 1);  ++tileIdx)
    {
        int i = hipThreadIdx_y;
        int j = hipThreadIdx_x;
	//int m = hipThreadIdx_y * BLOCK_SIZE1 + hipThreadIdx_x;
        if (tileIdx * BLOCK_SIZE1 + j < K && col + i < N) 
        {
            Bs[j][i] = B[tileIdx * BLOCK_SIZE1 + j + (col + i) * ldb];
        } 
        else {
            Bs[j][i] = 0;
        }
	/*for(unsigned int idy = 0; idy < BLOCK_SIZE1 ; ++idy)
	{
	    if (tileIdx *BLOCK_SIZE1 + idy < K && rc + m < M)
	    {
	        As[idy][m] = A[(tileIdx * BLOCK_SIZE1 + idy) * lda + rc + m];
	    }
	    else
	    {
	        As[idy][m] = 0;
	    }
	}*/
        __syncthreads();
        for (unsigned int idx = 0; idx < BLOCK_SIZE1; ++idx)
        {
	    //int m = hipThreadIdx_y * BLOCK_SIZE1 + hipThreadIdx_x;
            float a_reg;
	    if (row < M && tileIdx * BLOCK_SIZE1 + idx < K)
            {
                a_reg = A[row + (tileIdx * BLOCK_SIZE1 + idx) * lda];
            }
	    else
	    {
		a_reg = 0;
	    } 
            //a_reg = As[idx][m];
            #pragma unroll
            for (unsigned int outIdx = 0; outIdx < BLOCK_SIZE2; ++outIdx)
            {
                accumulator[outIdx] += a_reg * Bs[idx][outIdx];
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






    //     As[row][col] = A[i + (k + row) * lda];
    //     Bs[row][col] = B[k + col + j * ldb];
    //     __syncthreads();
    //     #pragma unroll
    //     for
    //     for (int warp_k = 0; warp_k < WARPNUM; warp_k++)
    //     {  
    //         int h = BLOCK_SIZE/WARPNUM;
    //         int thm=warp_k*h;
    //         #pragma unroll
    //         for (int thread_x = thm; thread_x < thm+h; ++thread_x)
    //         {
    //                 A_reg[col]=As[thread_x][col];
    //                 #pragma unroll
    //                 for (int thread_y=0; thread_y < BLOCK_SIZE; ++thread_y)
    //                 {
    //                     B_reg[row]=Bs[row][thread_y];
    //                     accumulator += A_reg[col]*B_reg[row];
    //                 }
    //         }
    //     }
    //     // for (int e = 0; e < BLOCK_SIZE;++e)
    //     //     accumulator += As[e][col] * Bs[row][e];
    //     __syncthreads();
    // }

    // if (res != 0)
    // {
    //     if (row < res)
    //         As[row][col] = A[i + (k + row) * lda];
    //     if (col < res)
    //         Bs[row][col] = B[k + col + j * ldb];

    //     __syncthreads();
    //     #pragma unroll
    //     for (int e = 0; e < res; ++e)
    //         accumulator += As[e][col] * Bs[row][e];
    //     __syncthreads();
    // }

    // if (i < M && j < N)
    // {
    //     C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    // }
