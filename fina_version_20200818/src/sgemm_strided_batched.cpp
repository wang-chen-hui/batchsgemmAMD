
#include "include/sgemm_strided_batched.h"
using namespace std;

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
                           int batch_count)
{
	cout << "compute sgemm stride batch" << endl;
}
