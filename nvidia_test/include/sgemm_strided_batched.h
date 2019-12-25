

/**/
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#define BLOCK_SIZE  16
typedef enum sgemm_operation_
{
    operation_none = 0,
    operation_transpose = 1,
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
    int col = threadIdx.x;
    int row = threadIdx.y;
    int i = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int j = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    if (i < M && j < N) 
    {
        float accumulator = 0;
        for (int k = 0; k < K; k +=BLOCK_SIZE)
        {
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];    
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    	    As[row][col] = A[i + (k + row) * lda];
	     	Bs[col][row] = B[k + col + j * ldb];
            __syncthreads();
            for (int e = 0; e < BLOCK_SIZE; ++e)
            accumulator += As[e][col] * Bs[e][row];
            __syncthreads(); 
        }
        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }

}

    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if (i < M && j < N)
    // {
    //     float accumulator = 0;

    //     for (int k = 0; k < K; ++k)
    //     {
    //         accumulator += A[i + k * lda] * B[k + j * ldb];
    //     }

    //     C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    // }
    //}

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

    if (trans_a == operation_none && trans_b == operation_none)
    {
        cout << "NO transpose" << endl;
    }
    else
    {
        cout << "Transpose!" << endl;
    }

    for (int i = 0; i < batch_count; i++)
    {
        cout << "Batched  : " << i << endl;
        ReferenceGemm_kernel<<<grid, block>>>(
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
