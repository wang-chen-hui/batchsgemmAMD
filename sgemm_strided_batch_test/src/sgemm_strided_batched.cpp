
#include "include/sgemm_strided_batched.h"
#include "hip/hip_runtime.h"
using namespace std;

__global__ void ReferenceGemm_kernel(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float *C, int ldc)
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
        C[i + j * ldc] = alpha * accumulator;
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
    dim3 block(16, 16);
    dim3 grid(
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    for (int i = 0; i < batch_count; i++)
    {
        A += stride_a * lda * i;
        B += stride_b * ldb * i;
        C += stride_c * ldc * i;
        hipLaunchKernelGGL(ReferenceGemm_kernel,
                           grid,
                           block,
                           0,
                           0,
                           m,
                           n,
                           k,
                           alpha,
                           A,
                           lda,
                           B,
                           ldb,
                           C,
                           ldc);
    }
}
