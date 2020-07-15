
/**/
#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include <iostream>
#include "hip/hip_runtime.h"
#include <time.h>
using namespace std;

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 16
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_X 4

typedef enum sgemm_operation_
{
    operation_none = 0,
    operation_transpose = 1,
    operation_conjugate_transpose = 2
} sgemm_operation;



// cal o
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void ReferenceGemm_kernel( 
    int M,
    int N,
    int K,
    float alpha,
    float *  __restrict__ A,
    int lda,
    int stride_a,
    float * __restrict__ B,
    int ldb,
    int stride_b,
    float beta,
    float * __restrict__ C,
    int ldc,
    int stride_c) {

    int batch_id = hipBlockIdx_z;
    A += stride_a * batch_id;
    B += stride_b * batch_id;
    C += stride_c * batch_id;

    // Block index
    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;

    // Thread index
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    
    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
    
    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL + BLOCK_SIZE_N * bx, // col
                    N )]);
        }
    
        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
            }
            
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)] = alpha * accum[thread_y][thread_x] + beta * 
            C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]
                ;
        }
    }
}



void sgemm_strided_batched2(sgemm_operation trans_a,
                           sgemm_operation trans_b,
                           int M,
                           int N,
                           int K,
                           const float *alpha,
                           float *A,
                           int lda,
                           int stride_a,
                           float *B,
                           int ldb,
                           int stride_b,
                           const float *beta,
                           float *C,
                           int ldc,
                           int stride_c,
                           int batch_count)
{
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M, batch_count);

    hipLaunchKernelGGL(ReferenceGemm_kernel, dimGrid, dimBlock, 0, 0,
                            M,
                            N,
                            K,
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

void sgemm_strided_batched(sgemm_operation trans_a,
                           sgemm_operation trans_b,
                           int m,
                           int n,
                           int k,
                           const float *alpha,
                           float *A,
                           int lda,
                           int stride_a,
                           float *B,
                           int ldb,
                           int stride_b,
                           const float *beta,
                           float *C,
                           int ldc,
                           int stride_c,
                           int batch_count)
{
     sgemm_strided_batched2(trans_a,
                            trans_b,
                            n,
                            m,
                            k,
                            alpha,
                            B,
                            k,
                            stride_b,
                            A,
                            m,
                            stride_a,
                            beta,
                            C,
                            m,
                            stride_c,
                            batch_count);
}
#endif
