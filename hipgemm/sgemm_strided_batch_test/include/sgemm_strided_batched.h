
/**/
#include <iostream>

#ifndef SGEMM_STRIDED_BATCHED_
#define SGEMM_STRIDED_BATCHED_

typedef enum sgemm_operation_ {
    operation_none      = 0, 
    operation_transpose  = 1,
    operation_conjugate_transpose = 2
} sgemm_operation;



void sgemm_strided_batched(sgemm_operation trans_a,
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
                           int batch_count);
#endif
