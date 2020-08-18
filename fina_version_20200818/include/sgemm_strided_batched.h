
/**/
#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

#include <iostream>
#include "hip/hip_runtime.h"
#include <time.h>
using namespace std;

#define BLOCK_SIZE_M_ 128
#define BLOCK_SIZE_N_ 64
#define BLOCK_SIZE_K_ 16
#define THREAD_SIZE_Y_ 8
#define THREAD_SIZE_X_ 4

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
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// inline __device__ static void mad(
//     float &d,
//     const float &a,
//     const float &b,
//     const float &c)
// {
//     d = a * b + c;
//     // asm volatile("fma.rn.f32 %0, %1, %2, %3;\n"
//     //              : "=f"(d)
//     //              : "f"(a), "f"(b), "f"(c));
// }

// inline __device__ void mad_xy(
//     float (&accumulators)[THREAD_SIZE_Y][THREAD_SIZE_X],
//     float (&tile_a)[THREAD_SIZE_Y],
//     float (&tile_b)[THREAD_SIZE_X],
//     int x,
//     int y)
// {
//     mad(
//         accumulators[y][x],
//         tile_a[y],
//         tile_b[x],
//         accumulators[y][x]);
// }

template<int THREAD_SIZE_Y, int THREAD_SIZE_X>
inline __device__ void multiply_accumulate(
    float (&accumulators)[THREAD_SIZE_Y][THREAD_SIZE_X],
    float (&tile_a)[THREAD_SIZE_Y],
    float (&tile_b)[THREAD_SIZE_X])
{
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

template <bool isPadding,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X,
    int THREAD_NUM_PER_BLOCK,
    int A_TILE_THREAD_PER_ROW,
    int B_TILE_THREAD_PER_ROW,
    int A_TILE_ROW_STRIDE,
    int B_TILE_ROW_STRIDE
>
inline __device__ void global_to_shared_fetch_fast(
    float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    float *A,
    float *B,
    const int M,
    const int N,
    const int K,
    const int by,
    const int bx,
    const int tile_id,
    const int tile_idx,
    const int A_TILE_ROW_START,
    const int B_TILE_ROW_START,
    const int A_TILE_COL,
    const int B_TILE_COL)
{
    // load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        if (isPadding)
        {
            // if (BLOCK_SIZE_M * by + A_TILE_ROW_START + i < M && A_TILE_COL + tile_idx < K)
            (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_M][A_TILE_COL]) = (A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx,                    // col
                K)]);
        }
        else
        {
            (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_M][A_TILE_COL]) = (A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx,                    // col
                K)]);
        }
    }

// load B from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        if (isPadding)
        {
            // if (B_TILE_ROW_START + i + tile_idx < K && B_TILE_COL + BLOCK_SIZE_N * bx < N + 4)
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i + tile_idx, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);
        }
        else
        {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i + tile_idx, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);
        }
    }
}


template <
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X
>
inline __device__ void shared_to_register_fetch_fast(
    float frag_a[THREAD_SIZE_Y],
    float frag_b[THREAD_SIZE_X],
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
        (frag_a[thread_y]) = (As[ty * THREAD_SIZE_Y + thread_y + tidx * BLOCK_SIZE_M][kidx]);
    }

#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
    {
        FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[kidx + tidx * BLOCK_SIZE_K][THREAD_SIZE_X * tx + thread_x]);
    }
}

template <
    bool isPadding,
    sgemm_operation trans_a,
    sgemm_operation trans_b,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X,
    int THREAD_NUM_PER_BLOCK,
    int A_TILE_THREAD_PER_ROW,
    int B_TILE_THREAD_PER_ROW,
    int A_TILE_ROW_STRIDE,
    int B_TILE_ROW_STRIDE
>
inline __device__ void global_to_shared_fetch(
    float As[BLOCK_SIZE_K * 2][BLOCK_SIZE_M],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    float *A,
    float *B,
    const int M,
    const int N,
    const int K,
    const int by,
    const int bx,
    const int tile_id,
    const int tile_idx,
    const int A_TILE_ROW_START,
    const int B_TILE_ROW_START,
    const int A_TILE_COL,
    const int B_TILE_COL)   
{
    if(trans_a == operation_none)
    {
        // load A from global memory to shared memory
    #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_ROW_STRIDE)
        {
            (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][A_TILE_COL]) = (A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_COL, // row
                A_TILE_ROW_START + i + tile_idx,                    // col
                K)]);
        }
    }
    else if(trans_a == operation_transpose)
    {
        // load A from global memory to shared memory
    #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += A_TILE_ROW_STRIDE)
        {
            (As[A_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][A_TILE_COL]) = (A[OFFSET(
                A_TILE_ROW_START + i + tile_idx, // row
                BLOCK_SIZE_M * by + A_TILE_COL,                    // col
                M)]);
        }
    }


    if(trans_b == operation_none)
    {
    // load B from global memory to shared memory
    #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
        {
            (Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = (B[OFFSET(
                B_TILE_ROW_START + i + tile_idx, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);
        }
    }
    else if(trans_b == operation_transpose)
    {
    // load B from global memory to shared memory
    #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
        {
            (Bs[B_TILE_ROW_START + i + (tile_id % 2) * BLOCK_SIZE_K][B_TILE_COL]) = (B[OFFSET(
                B_TILE_COL + BLOCK_SIZE_N * bx, // row
                B_TILE_ROW_START + i + tile_idx,  // col
                K)]);
        }

    }
}

template <
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X
>
inline __device__ void shared_to_register_fetch(
    float frag_a[THREAD_SIZE_Y],
    float frag_b[THREAD_SIZE_X],
    float As[BLOCK_SIZE_K * 2][BLOCK_SIZE_M],
    float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    int kidx,
    int tidx,
    int ty,
    int tx)
{
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
    {
        FETCH_FLOAT4(frag_a[thread_y]) = FETCH_FLOAT4(As[kidx + tidx * BLOCK_SIZE_K][ty * THREAD_SIZE_Y + thread_y]);
    }

#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
    {
        FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[kidx + tidx * BLOCK_SIZE_K][THREAD_SIZE_X * tx + thread_x]);
    }
}

template <
    bool isPadding,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X
>
inline __device__ void epilogue(
    float *C,
    float (&accumulators)[THREAD_SIZE_Y][THREAD_SIZE_X],
    float Cs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N],
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int by,
    int bx,
    int ty,
    int tx,
    int tid)
{
    int idx = 0;
    if (isPadding)
    {
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
            {
                if (BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y < M && BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x < N)
                    C[OFFSET(
                        BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                        BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                        N)] = alpha * accumulators[thread_y][thread_x] + beta *
                    C[OFFSET(
                        BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                        BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                        N)];
            }
        }
    }
    else
    {        
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
            {
                    C[OFFSET(
                        BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                        BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                        N)] = alpha * accumulators[thread_y][thread_x] + beta *
                    C[OFFSET(
                        BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                        BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                        N)];
            }
        }

    }
}

template <
    bool isPadding,
    sgemm_operation trans_a,
    sgemm_operation trans_b,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X
>
__global__ void ReferenceGemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float *__restrict__ A,
    int lda,
    int stride_a,
    float *__restrict__ B,
    int ldb,
    int stride_b,
    float beta,
    float *__restrict__ C,
    int ldc,
    int stride_c)
{
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

    // Size of thread block.
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // Thread id
    const int tid = ty * bszx + tx;

    // Shared memory
    __shared__ float As[BLOCK_SIZE_K * 2][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N];

    // Registers
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];


    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_M / 1;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 1;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 1;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 1;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    int res_size = BLOCK_SIZE_K;
    int res_tile_id = K / BLOCK_SIZE_K - 1;

    if (isPadding)
    {
        if (K % BLOCK_SIZE_K != 0)
        {
            res_size = K % BLOCK_SIZE_K;
            res_tile_id += 1;
        }
    }

    global_to_shared_fetch<
        isPadding,
        trans_a,
        trans_b,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X,
        THREAD_NUM_PER_BLOCK,
        A_TILE_THREAD_PER_ROW,
        B_TILE_THREAD_PER_ROW,
        A_TILE_ROW_STRIDE,
        B_TILE_ROW_STRIDE
    >(As, Bs, A, B, M, N, K, by, bx, 0, 0,
                                      A_TILE_ROW_START,
                                      B_TILE_ROW_START,
                                      A_TILE_COL,
                                      B_TILE_COL);
    __syncthreads();

    for (int tile_idx = BLOCK_SIZE_K; tile_idx < K; tile_idx += BLOCK_SIZE_K)
    {
        int tile_id = tile_idx / BLOCK_SIZE_K - 1;

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {

            shared_to_register_fetch<
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            THREAD_SIZE_Y,
            THREAD_SIZE_X>(frag_a, frag_b, As, Bs, k, tile_id % 2, ty, tx);

            multiply_accumulate(accum, frag_a, frag_b);
        }

        global_to_shared_fetch<
        isPadding,
        trans_a,
        trans_b,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X,
        THREAD_NUM_PER_BLOCK,
        A_TILE_THREAD_PER_ROW,
        B_TILE_THREAD_PER_ROW,
        A_TILE_ROW_STRIDE,
        B_TILE_ROW_STRIDE>(As, Bs, A, B, M, N, K, by, bx, tile_id + 1, tile_idx,
                                      A_TILE_ROW_START,
                                      B_TILE_ROW_START,
                                      A_TILE_COL,
                                      B_TILE_COL);

        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < res_size; ++k)
    {

        shared_to_register_fetch<
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            THREAD_SIZE_Y,
            THREAD_SIZE_X>(frag_a, frag_b, As, Bs, k, res_tile_id % 2, ty, tx);

        multiply_accumulate<THREAD_SIZE_Y, THREAD_SIZE_X>(accum, frag_a, frag_b);
    }

    epilogue<
        isPadding,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X>(C, accum, Bs, M, N, K, alpha, beta, by, bx, ty, tx, tid);
}

template <
    bool isPadding,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int THREAD_SIZE_Y,
    int THREAD_SIZE_X
>
__global__ void ReferenceGemm_kernel_fast(
    int M,
    int N,
    int K,
    float alpha,
    float *__restrict__ A,
    int lda,
    int stride_a,
    float *__restrict__ B,
    int ldb,
    int stride_b,
    float beta,
    float *__restrict__ C,
    int ldc,
    int stride_c)
{
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

    // Size of thread block.
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // Thread id
    const int tid = ty * bszx + tx;

    // Shared memory

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

    int res_size = BLOCK_SIZE_K;
    int res_tile_id = K / BLOCK_SIZE_K - 1;

    if (isPadding)
    {
        if (K % BLOCK_SIZE_K != 0)
        {
            res_size = K % BLOCK_SIZE_K;
            res_tile_id += 1;
        }
    }

    global_to_shared_fetch_fast<
        isPadding,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X,
        THREAD_NUM_PER_BLOCK,
        A_TILE_THREAD_PER_ROW,
        B_TILE_THREAD_PER_ROW,
        A_TILE_ROW_STRIDE,
        B_TILE_ROW_STRIDE
    >(As, Bs, A, B, M, N, K, by, bx, 0, 0,
                                      A_TILE_ROW_START,
                                      B_TILE_ROW_START,
                                      A_TILE_COL,
                                      B_TILE_COL);
    __syncthreads();

    for (int tile_idx = BLOCK_SIZE_K; tile_idx < K; tile_idx += BLOCK_SIZE_K)
    {
        int tile_id = tile_idx / BLOCK_SIZE_K - 1;

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {

            shared_to_register_fetch_fast<
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            THREAD_SIZE_Y,
            THREAD_SIZE_X>(frag_a, frag_b, As, Bs, k, tile_id % 2, ty, tx);

            multiply_accumulate(accum, frag_a, frag_b);
        }

        global_to_shared_fetch_fast<
        isPadding,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X,
        THREAD_NUM_PER_BLOCK,
        A_TILE_THREAD_PER_ROW,
        B_TILE_THREAD_PER_ROW,
        A_TILE_ROW_STRIDE,
        B_TILE_ROW_STRIDE>(As, Bs, A, B, M, N, K, by, bx, tile_id + 1, tile_idx,
                                      A_TILE_ROW_START,
                                      B_TILE_ROW_START,
                                      A_TILE_COL,
                                      B_TILE_COL);

        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < res_size; ++k)
    {

        shared_to_register_fetch_fast<
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            THREAD_SIZE_Y,
            THREAD_SIZE_X>(frag_a, frag_b, As, Bs, k, res_tile_id % 2, ty, tx);

        multiply_accumulate<THREAD_SIZE_Y, THREAD_SIZE_X>(accum, frag_a, frag_b);
    }

    epilogue<
        isPadding,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X>(C, accum, Bs, M, N, K, alpha, beta, by, bx, ty, tx, tid);
}

typedef __global__ void (*KERNEL)(
    int M,
    int N,
    int K,
    float alpha,
    float *__restrict__ A,
    int lda,
    int stride_a,
    float *__restrict__ B,
    int ldb,
    int stride_b,
    float beta,
    float *__restrict__ C,
    int ldc,
    int stride_c);

template <
    sgemm_operation trans_a,
    sgemm_operation trans_b
>
void sgemm_strided_batched1(int M,
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
    if(M == 64)
    {
        dim3 dimBlock(16 , 16);
        dim3 dimGrid((N + 63) / 64, (M + 15) / 16, batch_count);


        KERNEL kernel = ReferenceGemm_kernel<
        true,
        trans_a,
        trans_b,
        16,
        64,
        16,
        1,
        4>;
        if (M % 16 == 0 && N % 64 == 0 && K % 16 == 0)
        {
            kernel = ReferenceGemm_kernel<
            false,
            trans_a,
            trans_b,
            16,
            64,
            16,
            1,
            4>;
        }
            hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
    else if(M == 128)
    {
        dim3 dimBlock(16 , 16);
        dim3 dimGrid((N + 63) / 64, (M + 31) / 32, batch_count);


        KERNEL kernel = ReferenceGemm_kernel<
        true,
        trans_a,
        trans_b,
        32,
        64,
        16,
        2,
        4>;
        if (M % 32 == 0 && N % 64 == 0 && K % 16 == 0)
        {
            kernel = ReferenceGemm_kernel<
            false,
            trans_a,
            trans_b,
            32,
            64,
            16,
            2,
            4>;
        }
            hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
    else
    {
    dim3 dimBlock(BLOCK_SIZE_N_ / THREAD_SIZE_X_, BLOCK_SIZE_M_ / THREAD_SIZE_Y_);
    dim3 dimGrid((N + BLOCK_SIZE_N_ - 1) / BLOCK_SIZE_N_, (M + BLOCK_SIZE_M_ - 1) / BLOCK_SIZE_M_, batch_count);

    KERNEL kernel = ReferenceGemm_kernel<
        true,
        trans_a,
        trans_b,
        BLOCK_SIZE_M_,
        BLOCK_SIZE_N_,
        BLOCK_SIZE_K_,
        THREAD_SIZE_Y_,
        THREAD_SIZE_X_>;
    if (M % BLOCK_SIZE_M_ == 0 && N % BLOCK_SIZE_N_ == 0 && K % BLOCK_SIZE_K_ == 0)
    {
        kernel = ReferenceGemm_kernel<
        false,
        trans_a,
        trans_b,
        BLOCK_SIZE_M_,
        BLOCK_SIZE_N_,
        BLOCK_SIZE_K_,
        THREAD_SIZE_Y_,
        THREAD_SIZE_X_>;
    }
    hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
}

void sgemm_strided_batched2(int M,
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
    if(M == 64)
    {
        dim3 dimBlock(16 , 16);
        dim3 dimGrid((N + 63) / 64, (M + 15) / 16, batch_count);

        KERNEL kernel = ReferenceGemm_kernel_fast<
        true,
        16,
        64,
        16,
        1,
        4>;
        if (M % 16 == 0 && N % 64 == 0 && K % 16 == 0)
        {
            kernel = ReferenceGemm_kernel_fast<
            false,
            16,
            64,
            16,
            1,
            4>;
        }
            hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
    else if(M == 128)
    {
        dim3 dimBlock(16 , 16);
        dim3 dimGrid((N + 63) / 64, (M + 31) / 32, batch_count);


        KERNEL kernel = ReferenceGemm_kernel_fast<
        true,
        32,
        64,
        16,
        2,
        4>;
        if (M % 32 == 0 && N % 64 == 0 && K % 16 == 0)
        {
            kernel = ReferenceGemm_kernel_fast<
            false,
            32,
            64,
            16,
            2,
            4>;
        }
            hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
    else
    {
    dim3 dimBlock(BLOCK_SIZE_N_ / THREAD_SIZE_X_, BLOCK_SIZE_M_ / THREAD_SIZE_Y_);
    dim3 dimGrid((N + BLOCK_SIZE_N_ - 1) / BLOCK_SIZE_N_, (M + BLOCK_SIZE_M_ - 1) / BLOCK_SIZE_M_, batch_count);


    KERNEL kernel = ReferenceGemm_kernel_fast<
        true,
        BLOCK_SIZE_M_,
        BLOCK_SIZE_N_,
        BLOCK_SIZE_K_,
        THREAD_SIZE_Y_,
        THREAD_SIZE_X_>;
    if (M % BLOCK_SIZE_M_ == 0 && N % BLOCK_SIZE_N_ == 0 && K % BLOCK_SIZE_K_ == 0)
    {
        kernel = ReferenceGemm_kernel_fast<
        false,
        BLOCK_SIZE_M_,
        BLOCK_SIZE_N_,
        BLOCK_SIZE_K_,
        THREAD_SIZE_Y_,
        THREAD_SIZE_X_>;
    }
    hipLaunchKernelGGL(kernel, dimGrid, dimBlock, 0, 0,
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
    if(trans_a == operation_none && trans_b == operation_none)
    {
        sgemm_strided_batched2(
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
    else if(trans_a == operation_transpose && trans_b == operation_none)
    {
        sgemm_strided_batched1<operation_none, operation_transpose>(
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
    else if(trans_a == operation_none && trans_b == operation_transpose)
    {
        sgemm_strided_batched1<operation_transpose, operation_none>(
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
    else if(trans_a == operation_transpose && trans_b == operation_transpose)
    {
        sgemm_strided_batched1<operation_transpose, operation_transpose>(
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
    

}
#endif
