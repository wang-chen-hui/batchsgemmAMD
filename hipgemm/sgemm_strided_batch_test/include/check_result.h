#include <cstdlib>
#include <cstdio>
#include <vector>
#include <limits>
#include <iostream>
#include <string>

using std::vector;

#ifndef CHECK_RESULT_
#define CHECK_RESULT_


bool check_result(vector<float>& ha,vector<float>& hb,vector<float>& hc,vector<float>& hc_gold,float alpha,float beta,int m,int n,int k,int batch_count,int stride_a,int a_stride_1,int a_stride_2,int stride_b,int b_stride_1,int b_stride_2,int stride_c,int ldc,int size_c);

#endif

