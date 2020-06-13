/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <stdio.h>
#include <string.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;



#include <cstdlib>
#include <cstdio>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <hip/hip_runtime.h>
using std::vector;

#include "include/check_result.h"
#include "include/sgemm_strided_batched.h"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif


// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10
#define ALPHA 2
#define BETA 3


void printMatrix(const char* name, float* A, int m, int n, int lda)
{
    printf("---------- %s ----------\n", name);
    int max_size = 15;
    for(int i = 0; i < m && i < max_size; i++)
    {
        for(int j = 0; j < n && j < max_size; j++)
        {
            printf("%f ", A[i + j * lda]);
        }
        printf("\n");
    }
}

void print_strided_batched(const char* name,
                           float* A,
                           int n1,
                           int n2,
                           int n3,
                           int s1,
                           int s2,
                           int s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s ----------\n", name);
    int max_size = 3;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("%8.1f ", A[(i1 * s1) + (i2 * s2) + (i3 * s3)]);
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}



static void show_usage(char* argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-m \t\t\tm\t\tGEMM_STRIDED_BATCHED argument m\n"
              << "\t-n \t\t\tn\t\tGEMM_STRIDED_BATCHED argument n\n"
              << "\t-k \t\t\tk \t\tGEMM_STRIDED_BATCHED argument k\n"
              << "\t--lda \t\t\tlda \t\tGEMM_STRIDED_BATCHED argument lda\n"
              << "\t--ldb \t\t\tldb \t\tGEMM_STRIDED_BATCHED argument ldb\n"
              << "\t--ldc \t\t\tldc \t\tGEMM_STRIDED_BATCHED argument ldc\n"
              << "\t--stride_a \t\tstride_a \tGEMM_STRIDED_BATCHED argument stride_a\n"
              << "\t--stride_b \t\tstride_b \tGEMM_STRIDED_BATCHED argument stride_b\n"
              << "\t--stride_c \t\tstride_c \tGEMM_STRIDED_BATCHED argument stride_c\n"
              << "\t--batch_count \t\tbatch_count \tGEMM_STRIDED_BATCHED argument batch count\n"
              << "\t--alpha \t\talpha \t\tGEMM_STRIDED_BATCHED argument alpha\n"
              << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED_BATCHED argument beta\n"
              << "\t--header \t\theader \t\tprint header for output\n"
              << std::endl;
}

static int parse_arguments(int argc,
                           char* argv[],
                           int& m,
                           int& n,
                           int& k,
                           int& lda,
                           int& ldb,
                           int& ldc,
                           int& stride_a,
                           int& stride_b,
                           int& stride_c,
                           int& batch_count,
                           float& alpha,
                           float& beta,
						   sgemm_operation& trans_a,
						   sgemm_operation& trans_b)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "-k") && (i + 1 < argc))
                {
                    k = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb = atoi(argv[++i]);
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc = atoi(argv[++i]);
                }
                else if((arg == "--stride_a") && (i + 1 < argc))
                {
                    stride_a = atoi(argv[++i]);
                }
                else if((arg == "--stride_b") && (i + 1 < argc))
                {
                    stride_b = atoi(argv[++i]);
                }
                else if((arg == "--stride_c") && (i + 1 < argc))
                {
                    stride_c = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = atof(argv[++i]);
                }
				else if((arg == "--trans_a") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_b") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                       trans_b = operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

bool bad_argument(sgemm_operation trans_a,
                  sgemm_operation trans_b,
                  int m,
                  int n,
                  int k,
                  int lda,
                  int ldb,
                  int ldc,
                  int stride_a,
                  int stride_b,
                  int stride_c,
                  int batch_count)
{	
   bool argument_error = false;
   if((trans_a == operation_none) && (lda < m))
   {
      argument_error = true;
      std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
   }
   if((trans_a == operation_transpose) && (lda < k))
   {
      argument_error = true;
      std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
   }
   if((trans_b == operation_none) && (ldb < k))
   {
      argument_error = true;
      std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
   }
   if((trans_b == operation_transpose) && (ldb < n))
   {
      argument_error = true;
      std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << n << std::endl;
   }
   if(stride_a < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_a < 0" << std::endl;
    }
    if(stride_b < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_b < 0" << std::endl;
    }
    if(ldc < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldc << " < " << m << std::endl;
    }
    if(stride_c < n * ldc)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_c << " < " << n * ldc << std::endl;
    }
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

void initialize_a_b_c(vector<float>& ha,
                      int size_a,
                      vector<float>& hb,
                      int size_b,
                      vector<float>& hc,
                      vector<float>& hc_gold,
                      int size_c)
{
    srand(size_a);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
       //      ha[i] = i;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 5;
        //      hb[i] = 1.0;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 3;
        //      hc[i] = 1.0;
    }
    hc_gold = hc;
}


void matrix_read(vector<float>& hm,int size,const char* file_matrix)
{
	std::ifstream file_m;
	std::string line;
	file_m.open(file_matrix);
	int kk = 0;
	if (file_m.is_open())
	{
		while(size > 0){
			getline(file_m,line);
			std::stringstream is(line);
			int ele;
			while(is >> ele){
				hm[kk] = ele/1.0;
				kk++;
			}
		size--;
		}
	}
	//std::cout << hm[0] << std::endl;
}


const char *publicKey = " \n\
-----BEGIN PUBLIC KEY----- \n\
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCvxN0avcOTfP+DYyQYgm+YkiqZ \n\
7bypafEJjXSKC9ardGd542ltIUd3um68m6LtSnj8xqinDZNdFfgSIvg5BAKoDJC9 \n\
1JC4UUhu8LBkVQJYgDddDUv6FL9YH7/YG/PPe6jl0661pX+9cWOhE2Mo4Fo6bevz \n\
D6KNjr0KXHRajs0m2wIDAQAB \n\
-----END PUBLIC KEY----- \
";


int resultencrypt(unsigned char * data, int datalength, unsigned char ** encryptdata, int * encryptlength)
{
	BIO *bio = NULL;
	RSA *publicRsa = NULL;
	int retcode = 0;
	int key_len = 0;

	if (data == NULL || datalength == 0){
		printf("Error on reading Data.");
		return -2;
	}

	bio = BIO_new_mem_buf((void *)publicKey, -1);
	if (bio == NULL) {
        printf("Memory Allocation Error\n");
        return -2;
    }

 	publicRsa = PEM_read_bio_RSA_PUBKEY(bio, NULL, NULL, NULL);
    if (publicRsa == NULL) {
        printf("Key Reader Error\n");
        return -2;
    }

    BIO_free_all(bio);
	
	key_len = RSA_size(publicRsa);
 
	unsigned char *encryptMsg = (unsigned char *)malloc(key_len);
	memset(encryptMsg, 0, key_len);

	if (key_len < datalength){
    	printf("Out of memory.\n");
        return -2;
    }
 	
	retcode = RSA_public_encrypt(datalength, data, encryptMsg, publicRsa, RSA_PKCS1_PADDING);

	if (retcode < 0){
		printf("Encrypt Error!\n");
		return -1;
	}
	*encryptdata = encryptMsg;
	*encryptlength = key_len;
	RSA_free(publicRsa);
	return 0;
}

void Char2Hex(unsigned char ch, unsigned char *szHex)
{
    int i;
    unsigned char byte[2];
    byte[0] = ch/16;
    byte[1] = ch%16;
    for(i=0; i<2; i++)
    {
        if(byte[i] >= 0 && byte[i] <= 9)
            szHex[i] = '0' + byte[i];
        else
            szHex[i] = 'A' + byte[i] - 10;
    }
}

unsigned char * CharStr2HexStr(unsigned char *pucCharStr, int iSize)
{
    int i;
    unsigned char szHex[3];
    unsigned char *pszHexStr = NULL;
    pszHexStr = (unsigned char *) malloc(2*iSize*sizeof(unsigned char)+1);

    for(i=0; i<iSize; i++)
    {
        Char2Hex(pucCharStr[i], szHex);
        pszHexStr[2*i] = szHex[0];
        pszHexStr[2*i+1] = szHex[1];
    }
    pszHexStr[2*iSize] = '\0';
    return pszHexStr;
}

int main(int argc, char* argv[])
{
    // initialize parameters with default values
    sgemm_operation trans_a = operation_none;
	sgemm_operation trans_b = operation_none;

    // invalid int and float for rocblas_sgemm_strided_batched int and float arguments
    int invalid_int = std::numeric_limits<int>::min() + 1;
    float invalid_float     = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    int m = invalid_int, lda = invalid_int, stride_a = invalid_int;
    int n = invalid_int, ldb = invalid_int, stride_b = invalid_int;
    int k = invalid_int, ldc = invalid_int, stride_c = invalid_int;

    int batch_count = invalid_int;

    float alpha = invalid_float;
    float beta  = invalid_float;
    
	const char* matrix_a = "./data/matrix_a.txt";
	const char* matrix_b = "./data/matrix_b.txt";
	const char* matrix_c = "./data/matrix_c.txt";
 
    //clock
    hipEvent_t start,end;
    hipEventCreate(&start);
    hipEventCreate(&end);    

    if(parse_arguments(argc,
                       argv,
                       m,
                       n,
                       k,
                       lda,
                       ldb,
                       ldc,
                       stride_a,
                       stride_b,
                       stride_c,
                       batch_count,
                       alpha,
                       beta,
                       trans_a,
                       trans_b))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(m == invalid_int)
        m = DIM1;
    if(n == invalid_int)
        n = DIM2;
    if(k == invalid_int)
        k = DIM3;
    if(lda == invalid_int)
		lda = trans_a == operation_none ? m : k;
    if(ldb == invalid_int)
		ldb = trans_b == operation_none ? k : n;
    if(ldc == invalid_int)
        ldc = m;
    if(stride_a == invalid_int)
		stride_a = trans_a == operation_none ? lda * k : lda * m;
    if(stride_b == invalid_int)
		stride_b = trans_b == operation_none ? ldb * n : ldb * k;
    if(stride_c == invalid_int)
        stride_c = ldc * n;
    if(alpha != alpha)
        alpha = ALPHA; // check for alpha == invalid_float == NaN
    if(beta != beta)
        beta = BETA; // check for beta == invalid_float == NaN
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(
           trans_a, trans_b, m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }


    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    int size_a1, size_b1, size_c1 = ldc * n;

    if(trans_a == operation_none)
    {
        //std::cout << "N";
        a_stride_1 = 1;
        a_stride_2 = lda;
        size_a1    = lda * k;
    }
    else
    {
        //std::cout << "T";
        a_stride_1 = lda;
        a_stride_2 = 1;
       size_a1    = lda * m;
    }
    if(trans_b == operation_none)
    {
        //std::cout << "N, ";
        b_stride_1 = 1;
        b_stride_2 = ldb;
        size_b1    = ldb * n;
    }
    else
    {
        //std::cout << "T, ";
        b_stride_1 = ldb;
        b_stride_2 = 1;
        size_b1    = ldb * k;
    }

/*
    std::cout << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << ", "
              << stride_a << ", " << stride_b << ", " << stride_c << ", " << batch_count << ", "
              << alpha << ", " << beta << ", ";
*/
    int size_a = batch_count == 0 ? size_a1 : size_a1 + stride_a * (batch_count - 1);
    int size_b = batch_count == 0 ? size_b1 : size_b1 + stride_b * (batch_count - 1);
    int size_c = batch_count == 0 ? size_c1 : size_c1 + stride_c * (batch_count - 1);

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    vector<float> ha(size_a);
    vector<float> hb(size_b);
    vector<float> hc(size_c);
    vector<float> hc_gold(size_c);

    // read data on host
	/*matrix_read(ha,size_a/size_a1,matrix_a);
	matrix_read(hb,size_b/size_b1,matrix_b);
	matrix_read(hc,size_c/size_c1,matrix_c);
	matrix_read(hc_gold,size_c/size_c1,matrix_c);*/
    //initial data on host
    initialize_a_b_c(ha, size_a, hb, size_b, hc, hc_gold, size_c);	

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));
    
    //time 
    float *d_time;
    CHECK_HIP_ERROR(hipMalloc(&d_time, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_time, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));
    
    //warmup && varify results
    sgemm_strided_batched(trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          &alpha,
                          da,
                          lda,
                          stride_a,
                          db,
                          ldb,
                          stride_b,
                          &beta,
                          dc,
                          ldc,
                          stride_c,
                          batch_count);

    double time=0.0;
    hipEventRecord(start,0);
    for(int s1 = 0; s1 < 10; ++s1)
    {
        sgemm_strided_batched(trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              &alpha,
                              da,
                              lda,
                              stride_a,
                              db,
                              ldb,
                              stride_b,
                              &beta,
                              d_time,
                              ldc,
                              stride_c,
                              batch_count);
    }

    hipEventRecord(end,0);
    hipEventSynchronize(end);
    float elapsed;
    hipEventElapsedTime(&elapsed,start,end);
    time += elapsed;
    time = time / 10;

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));
    
    bool result = check_result(ha,hb,hc,hc_gold,alpha,beta,m,n,k,batch_count,stride_a,a_stride_1,a_stride_2,stride_b,b_stride_1,b_stride_2,stride_c,ldc,size_c);
    std::ostringstream out;
    if ( result == 1 )
    {	
        out.precision(std::numeric_limits<double>::digits10);
        out << time;
        string time_output = out.str();
        unsigned char *source = (unsigned char *)time_output.c_str();
        cout << source << endl; 
        cout << time_output.length() << endl;

        unsigned char *encryptMsg = NULL;
        int encryptlength = 0;
        unsigned char * chartohexstring = NULL;
        
        resultencrypt(source, time_output.length(), &encryptMsg, &encryptlength);
        
        chartohexstring = CharStr2HexStr(encryptMsg, encryptlength);
        cout << "The result is  " << "\n" << chartohexstring << endl;
		
    }
    else
        std::cout << "Sorry,the result invaild" << std::endl;
	
    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIP_ERROR(hipFree(d_time));
    return EXIT_SUCCESS;
}
