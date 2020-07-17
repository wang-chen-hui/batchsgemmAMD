
/**/
#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include <iostream>
// #include "hip/hip_runtime.h"
#include <time.h>
using namespace std;

#define BLOCK_SIZE_M 128
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

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z


#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y



// cal o
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void MatrixMulCUDA6( 
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

    int stream_id = hipThreadIdx_z;

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

    __shared__ float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N];
    
    
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    
    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 1;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 1;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;


    if(stream_id == 1)
    {
        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            (As[A_TILE_ROW_START + i][A_TILE_COL]) = (A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL, // col
                    K )]);
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    B_TILE_ROW_START + i, // row
                    B_TILE_COL + BLOCK_SIZE_N * bx, // col
                    N )]);
        }
    }
    
        __syncthreads();
    // can not unroll since K can not be determined at this point
    for (int tile_idx = BLOCK_SIZE_K; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {


        int tile_id = tile_idx / BLOCK_SIZE_K;


        if(stream_id == 0)
        {
            // compute c
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
                // load A from shared memory to register
                #pragma unroll
                for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                    frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y + (tile_id - 1) % 2 * BLOCK_SIZE_M][k];
                }

                // load B from shared memory to register
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                    FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k + (tile_id - 1) % 2 * BLOCK_SIZE_K][THREAD_SIZE_X * tx + thread_x]);
                }
                
                #pragma unroll
                for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                    #pragma unroll
                    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                        accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                    }
                }
            }
        }

        if(stream_id == 1)
        {
            // load A from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_M][A_TILE_COL]) = (A[OFFSET(
                        BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                        A_TILE_COL + tile_idx, // col
                        K )]);
            }

            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                        tile_idx + B_TILE_ROW_START + i, // row
                        B_TILE_COL + BLOCK_SIZE_N * bx, // col
                        N )]);
            }
        }
        __syncthreads();
    }



    if(stream_id == 0)
    {
        int tile_id = K / BLOCK_SIZE_K;
        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y + (tile_id - 1) % 2 * BLOCK_SIZE_M][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k + (tile_id - 1) % 2 * BLOCK_SIZE_K][THREAD_SIZE_X * tx + thread_x]);
            }
            
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
        }


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

    // for(int tt = 0; tt < 4; tt++)
    // {
    //     if (tid / 64 == tt)
    //     {
    //         // store back to C
    //         #pragma unroll
    //         for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
    //             #pragma unroll
    //             for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
    //                 Bs[(ty % 4) * THREAD_SIZE_Y + thread_y][tx * THREAD_SIZE_X + thread_x] = alpha * accum[thread_y][thread_x];
    //             }
    //         }


    //         __syncthreads();
    //             #pragma unroll
    //         for (int ttt = 0; ttt < 32; ttt++)
    //         {
    //         C[OFFSET(BLOCK_SIZE_M * by + tt * 32 + ttt, BLOCK_SIZE_N * bx + tid % 64, N)]
    //         = Bs[ttt][tid % 64] + beta * 
    //         C[OFFSET(BLOCK_SIZE_M * by + tt * 32 + ttt, BLOCK_SIZE_N * bx + tid % 64, N)]
    //         ;
    //         }

    //     }
    //         __syncthreads();
    // }
}

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
    int idx = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int thread_num = hipBlockDim_x * hipBlockDim_y;

    int offset_a = hipBlockIdx_x * BLOCK_SIZE_M;
    int offset_b = hipBlockIdx_y * BLOCK_SIZE_N;
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_N][BLOCK_SIZE_K + 1];
    float accumulator = 0;
    int k = 0;
    int res = K % BLOCK_SIZE_K;

    for (k = 0; k < K - res; k += BLOCK_SIZE_K)
    {
        int idx_a_y = idx / BLOCK_SIZE_M;
        int idx_a_x = idx % BLOCK_SIZE_M;
        int k_num = thread_num / BLOCK_SIZE_M;        
        for(int k_iter = 0; k_iter < BLOCK_SIZE_K; k_iter += k_num)
        {
            int g_a_idx_y = k + k_iter + idx_a_y;
            int g_a_idx_x = idx_a_x + offset_a;

            if(g_a_idx_y < K && g_a_idx_x < M)
            As[k_iter + idx_a_y][idx_a_x] = A[g_a_idx_y * lda + g_a_idx_x];
            else
            {
                
            As[k_iter + idx_a_y][idx_a_x] = 0;
            }
            
        }

        int idx_b_y = idx / BLOCK_SIZE_K;
        int idx_b_x = idx % BLOCK_SIZE_K;
        int n_num = thread_num / BLOCK_SIZE_K;        
        for(int n_iter = 0; n_iter < BLOCK_SIZE_N; n_iter += n_num)
        {
            if(offset_b + idx_b_y + n_iter < N && k + idx_b_x < K)
                Bs[idx_b_y + n_iter][idx_b_x] = B[(offset_b + idx_b_y + n_iter) * ldb + k + idx_b_x];
            else
            {
                Bs[idx_b_y + n_iter][idx_b_x] = 0;
            }
            
        }

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE_K; ++e)
	        accumulator += As[e][hipThreadIdx_x] * Bs[hipThreadIdx_y][e];
        __syncthreads();
    }

    // if (res != 0)
    // {
    //     if (row < res)
    //         As[idx / BLOCK_SIZE_M][idx % BLOCK_SIZE_M] = A[i + (k + idx / BLOCK_SIZE_M) * lda];
    //     if (col < res)
    //         Bs[row][col] = B[k + col + j * ldb];
    //     __syncthreads();
    //     for (int e = 0; e < res; ++e)
	//         accumulator += As[e][col] * Bs[row][e];
    //     __syncthreads();
    // }

    // int idx_c_y = idx / BLOCK_SIZE_M;
    // int idx_c_x = idx % BLOCK_SIZE_M;
    // int n_num = thread_num / BLOCK_SIZE_M;        
    // for(int k_iter = 0; k_iter < BLOCK_SIZE_K; k_iter += k_num)
    // {
    //     C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    // }

    int i = offset_a + hipThreadIdx_x;
    int j = offset_b + hipThreadIdx_y;
    if (i < M && j < N)
    {
      C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    //   C[0] = beta * C[0];
    }
}

void sgemm_strided_batched1(sgemm_operation trans_a,
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
    dim3 block(BLOCK_SIZE_M, BLOCK_SIZE_N);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y,
        batch_count);


    ReferenceGemm_kernel<<<grid, block>>>(
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
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y,2);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M, batch_count);
        MatrixMulCUDA6<<< dimGrid, dimBlock >>>(
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
    //  sgemm_strided_batched1(trans_a,
    //                         trans_b,
    //                         m,
    //                         n,
    //                         k,
    //                         alpha,
    //                         A,
    //                         lda,
    //                         stride_a,
    //                         B,
    //                         ldb,
    //                         stride_b,
    //                         beta,
    //                         C,
    //                         ldc,
    //                         stride_c,
    //                         batch_count);

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
