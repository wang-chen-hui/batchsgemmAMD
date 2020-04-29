#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include<iostream>
#include "hip/hip_runtime.h"
#include <time.h> 

using namespace std;

#define BLOCK_SIZE 64
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

    int col = hipThreadIdx_x;
    int row = hipThreadIdx_y;
    int i = hipThreadIdx_x + hipBlockIdx_x * BLOCK_SIZE;
    int j = hipThreadIdx_y + hipBlockIdx_y * BLOCK_SIZE;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    float accumulator = 0;
    int k = 0;
    int res = K % BLOCK_SIZE;

    for (k = 0; k < K - res; k += BLOCK_SIZE)
    {

        As[row][col] = A[i + (k + row) * lda];
        Bs[row][col] = B[k + col + j * ldb];
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            accumulator += As[e][col] * Bs[row][e];
        __syncthreads();
    }

    if (res != 0)
    {
        if (row < res)
            As[row][col] = A[i + (k + row) * lda];
        if (col < res)
            Bs[row][col] = B[k + col + j * ldb];

        __syncthreads();
        for (int e = 0; e < res; ++e)
            accumulator += As[e][col] * Bs[row][e];
        __syncthreads();
    }

    if (i < M && j < N)
    {
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
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (m + block.x - 1) / block.x,
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