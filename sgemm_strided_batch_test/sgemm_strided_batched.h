
/**/
#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include <iostream>
#include "hip/hip_runtime.h"
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



// cal o
// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


inline __device__ static void mad(
    float &d,
    const float &a,
    const float &b,
    const float &c)
{
    d = a * b + c;
    // asm volatile("fma.rn.f32 %0, %1, %2, %3;\n"
    //              : "=f"(d)
    //              : "f"(a), "f"(b), "f"(c));
}

inline __device__ void mad_xy(
    float (&accumulators)[THREAD_SIZE_Y][THREAD_SIZE_X],
    float (&tile_a)[THREAD_SIZE_Y],
    float (&tile_b)[THREAD_SIZE_X],
    int x,
    int y)
{
    mad(
        accumulators[y][x],
        tile_a[y],
        tile_b[x],
        accumulators[y][x]);
}

inline __device__ void multiply_accumulate(
    float (&accumulators)[THREAD_SIZE_Y][THREAD_SIZE_X],
    float (&tile_a)[THREAD_SIZE_Y],
    float (&tile_b)[THREAD_SIZE_X])
{
// Simply traverse the accumulator tile in row-major order
#pragma unroll
    for (int y = 0; y < THREAD_SIZE_Y; ++y)
    {
#pragma unroll
        for (int x = 0; x < THREAD_SIZE_X; ++x)
        {
            accumulators[y][x] = tile_a[y] * tile_b[x] + accumulators[y][x];
        }
    }
}

inline __device__ void global_to_shared_request(
    float *frag_a,
    float *frag_b,
    float *A,
    float *B,
    int K,
    int N,
    int by,
    int bx,
    int tile_idx,
    const int THREAD_NUM_PER_BLOCK,
    const int A_TILE_THREAD_PER_ROW,
    const int B_TILE_THREAD_PER_ROW,
    const int A_TILE_ROW_START,
    const int B_TILE_ROW_START,
    const int A_TILE_COL,
    const int B_TILE_COL,
    const int A_TILE_ROW_STRIDE,
    const int B_TILE_ROW_STRIDE)
{
    // const int THREAD_NUM_PER_BLOCK = (BLOCK_SIZE_N / THREAD_SIZE_X) * (BLOCK_SIZE_M / THREAD_SIZE_Y);
    // // threads needed to load one row of tile
    // // / 4 is because float4 is used
    // const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 1;
    // const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // // row number and col number that needs to be loaded by this thread
    // const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    // const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 1;
    // const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // // row stride that thread uses to load multiple rows of a tile
    // const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        (frag_a[i / A_TILE_ROW_STRIDE * 1]) = (A[OFFSET(
            BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
            A_TILE_COL + tile_idx,                    // col
            K)]);
    }
// load B from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(frag_b[i / B_TILE_ROW_STRIDE * 4]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i + tile_idx, // row
            B_TILE_COL + BLOCK_SIZE_N * bx,  // col
            N)]);
    }
}

inline __device__ void global_to_shared_commit(
    float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    float *frag_a,
    float *frag_b,
    int tile_id,
    const int THREAD_NUM_PER_BLOCK,
    const int A_TILE_THREAD_PER_ROW,
    const int B_TILE_THREAD_PER_ROW,
    const int A_TILE_ROW_START,
    const int B_TILE_ROW_START,
    const int A_TILE_COL,
    const int B_TILE_COL,
    const int A_TILE_ROW_STRIDE,
    const int B_TILE_ROW_STRIDE
    )
{
    // const int THREAD_NUM_PER_BLOCK = (BLOCK_SIZE_N / THREAD_SIZE_X) * (BLOCK_SIZE_M / THREAD_SIZE_Y);
    // // threads needed to load one row of tile
    // // / 4 is because float4 is used
    // const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 1;
    // const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // // row number and col number that needs to be loaded by this thread
    // const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    // const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 1;
    // const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // // row stride that thread uses to load multiple rows of a tile
    // const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_M][A_TILE_COL]) = (frag_a[i / A_TILE_ROW_STRIDE * 1]);
    }

// load B from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(frag_b[i / B_TILE_ROW_STRIDE * 4]);
    }
}


inline __device__ void global_to_shared_fetch(
    float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    float *A,
    float *B,
    const int K,
    const int N,
    const int by,
    const int bx,
    const int tile_id,
    const int tile_idx,
    const int THREAD_NUM_PER_BLOCK,
    const int A_TILE_THREAD_PER_ROW,
    const int B_TILE_THREAD_PER_ROW,
    const int A_TILE_ROW_START,
    const int B_TILE_ROW_START,
    const int A_TILE_COL,
    const int B_TILE_COL,
    const int A_TILE_ROW_STRIDE,
    const int B_TILE_ROW_STRIDE
    )
{
    // const int THREAD_NUM_PER_BLOCK = (BLOCK_SIZE_N / THREAD_SIZE_X) * (BLOCK_SIZE_M / THREAD_SIZE_Y);
    // // threads needed to load one row of tile
    // // / 4 is because float4 is used
    // const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 1;
    // const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // // row number and col number that needs to be loaded by this thread
    // const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    // const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 1;
    // const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // // row stride that thread uses to load multiple rows of a tile
    // const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_M][A_TILE_COL]) = (A[OFFSET(
            BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
            A_TILE_COL + tile_idx,                    // col
            K)]);
    }

// load B from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i + tile_idx, // row
            B_TILE_COL + BLOCK_SIZE_N * bx,  // col
            N)]);
    }
}


inline __device__ void shared_to_register_fetch(
    // float frag_a[2][THREAD_SIZE_Y],
    // float frag_b[2][THREAD_SIZE_X],
    float frag_a[2][THREAD_SIZE_Y],
    float frag_b[2][THREAD_SIZE_X],
    float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    int kidx,
    int tidx,
    int ty,
    int tx)
{
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 1)
    {
        (frag_a[0][thread_y]) = (As[ty * THREAD_SIZE_Y + thread_y + tidx * BLOCK_SIZE_M][kidx]);
    }

#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
    {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[kidx + tidx * BLOCK_SIZE_K][THREAD_SIZE_X * tx + thread_x]);
    }

// #pragma unroll
//     for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 1)
//     {
//         (frag_a[kidx % 2][thread_y]) = (As[OFFSET(ty * THREAD_SIZE_Y + thread_y + tidx % 2 * BLOCK_SIZE_M, kidx, BLOCK_SIZE_K)]);
//     }

// #pragma unroll
//     for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
//     {
//         FETCH_FLOAT4(frag_b[kidx % 2][thread_x]) = FETCH_FLOAT4(Bs[OFFSET(kidx + tidx % 2 * BLOCK_SIZE_K, THREAD_SIZE_X * tx + thread_x, BLOCK_SIZE_N)]);
//     }
}

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

    __shared__ float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N];
    
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    
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


    float  frag_As[BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK];
    float  frag_Bs[BLOCK_SIZE_N * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK];
    
    // global_to_shared_request(frag_As, frag_Bs, A, B, K, N, by, bx, 0, 
    // THREAD_NUM_PER_BLOCK,
    // A_TILE_THREAD_PER_ROW,
    // B_TILE_THREAD_PER_ROW,
    // A_TILE_ROW_START,
    // B_TILE_ROW_START,
    // A_TILE_COL,
    // B_TILE_COL,
    // A_TILE_ROW_STRIDE,
    // B_TILE_ROW_STRIDE
    // );

    // // __syncthreads();

    // global_to_shared_commit(As, Bs, frag_As, frag_Bs, 0, 
    // THREAD_NUM_PER_BLOCK,
    // A_TILE_THREAD_PER_ROW,
    // B_TILE_THREAD_PER_ROW,
    // A_TILE_ROW_START,
    // B_TILE_ROW_START,
    // A_TILE_COL,
    // B_TILE_COL,
    // A_TILE_ROW_STRIDE,
    // B_TILE_ROW_STRIDE
    // );
    // __syncthreads();


// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//     {
//         (frag_As[i / A_TILE_ROW_STRIDE]) = (A[OFFSET(
//             BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
//             A_TILE_COL,                    // col
//             K)]);
//     }
// // load B from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//     {
//         FETCH_FLOAT4(frag_Bs[i / B_TILE_ROW_STRIDE * 4]) = FETCH_FLOAT4(B[OFFSET(
//             B_TILE_ROW_START + i, // row
//             B_TILE_COL + BLOCK_SIZE_N * bx,  // col
//             N)]);
//     }

// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//     {
//         (As[A_TILE_ROW_START + i][A_TILE_COL]) = (frag_As[i / A_TILE_ROW_STRIDE]);
//     }

// // load B from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//     {
//         FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(frag_Bs[i / B_TILE_ROW_STRIDE * 4]);
//     }
   
    global_to_shared_fetch(As, Bs, A, B, K, N, by, bx, 0, 0,
    THREAD_NUM_PER_BLOCK,
    A_TILE_THREAD_PER_ROW,
    B_TILE_THREAD_PER_ROW,
    A_TILE_ROW_START,
    B_TILE_ROW_START,
    A_TILE_COL,
    B_TILE_COL,
    A_TILE_ROW_STRIDE,
    B_TILE_ROW_STRIDE
    );
    __syncthreads();

    // can not unroll since K can not be determined at this point
    for (int tile_idx = BLOCK_SIZE_K; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        int tile_id = tile_idx / BLOCK_SIZE_K - 1;

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            
            shared_to_register_fetch(frag_a, frag_b, As, Bs, k, tile_id % 2, ty, tx);
            
        multiply_accumulate(accum, frag_a[0], frag_b[0]);
            
        }
        
        global_to_shared_fetch(As, Bs, A, B, K, N, by, bx, tile_id + 1, tile_idx,
        THREAD_NUM_PER_BLOCK,
        A_TILE_THREAD_PER_ROW,
        B_TILE_THREAD_PER_ROW,
        A_TILE_ROW_START,
        B_TILE_ROW_START,
        A_TILE_COL,
        B_TILE_COL,
        A_TILE_ROW_STRIDE,
        B_TILE_ROW_STRIDE
        );

        // global_to_shared_request(frag_As, frag_Bs, A, B, K, N, by, bx, tile_idx, 
        // THREAD_NUM_PER_BLOCK,
        // A_TILE_THREAD_PER_ROW,
        // B_TILE_THREAD_PER_ROW,
        // A_TILE_ROW_START,
        // B_TILE_ROW_START,
        // A_TILE_COL,
        // B_TILE_COL,
        // A_TILE_ROW_STRIDE,
        // B_TILE_ROW_STRIDE
        // );

        // global_to_shared_commit(As, Bs, frag_As, frag_Bs, tile_id + 1, 
        // THREAD_NUM_PER_BLOCK,
        // A_TILE_THREAD_PER_ROW,
        // B_TILE_THREAD_PER_ROW,
        // A_TILE_ROW_START,
        // B_TILE_ROW_START,
        // A_TILE_COL,
        // B_TILE_COL,
        // A_TILE_ROW_STRIDE,
        // B_TILE_ROW_STRIDE
        // );


// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//     {
//         float ta = (A[OFFSET(
//             BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
//             A_TILE_COL + tile_idx,                    // col
//             K)]);
        
//         (As[A_TILE_ROW_START + i + (tile_id + 1) % 2 * BLOCK_SIZE_M][A_TILE_COL]) = ta;
//     }
// // load B from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//     {
//         FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id + 1) % 2 * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
//             B_TILE_ROW_START + i + tile_idx, // row
//             BLOCK_SIZE_N * bx + B_TILE_COL,  // col
//             N)]);
        
//         // FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id + 1) % 2 * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(frag_Bs[i / B_TILE_ROW_STRIDE * 4]);
//     }

// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
//     {
//         (As[A_TILE_ROW_START + i + (tile_id + 1) % 2 * BLOCK_SIZE_M][A_TILE_COL]) = (frag_As[i / A_TILE_ROW_STRIDE]);
//     }

// // load B from global memory to shared memory
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
//     {
//         FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id + 1) % 2 * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(frag_Bs[i / B_TILE_ROW_STRIDE * 4]);
//     }
        __syncthreads();

        // __syncthreads();
    }


    int tile_id = K / BLOCK_SIZE_K - 1;
    // compute c
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; ++ k) {

        shared_to_register_fetch(frag_a, frag_b, As, Bs, k, tile_id % 2, ty, tx);
        
        multiply_accumulate(accum, frag_a[0], frag_b[0]);
        
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
