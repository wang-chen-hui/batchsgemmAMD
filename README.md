# batchsgemmAMD
use hip and rocm write batchsgemm code
## commit log 
+ > first commit 
  >> local commit to github
+ > service test
  >> service commit to github
## hiptest.cpp
use ___hiptest.cpp___ to test ```hipcc -name``` is or not can be run.
## hipgemm.cpp
basic gemm code use hip.
## hipcublas.cpp
cublas use hip to write.
## test.slurm
``` bash
sbtach test.slurm
```
## result 
solution

## sgemm_test_by_authorize
1. ___makefile___ has some problem,so we need to use ```LIB_PATH = lib/libcheckresult.so lib/libsgemm_strided_batched.so -lcrypto``` to replace ```LIB_PATH = -L./lib -lcheckresult -lsgemm_strided_batched -lcrypto```
2. or we also can complie this code by ```hipcc example_sgemm_strided_batched.cpp -I./include lib/libcheckresult.so lib/libsgemm_strided_batched.so -lcrypto -w -o example_sgemm_strided_batched```

## Currently
this projiect only has hip code

## SgemmStridedBatched
1. 此函数执行一批矩阵的矩阵矩阵乘法。批处理被认为是“统一的”，即所有实例的A，B和C矩阵都具有相同的尺寸（m，n，k），前导尺寸（lda，ldb，ldc）和换位（transa，transb） 。批次的每个实例的输入矩阵A，B和输出矩阵C与前一个实例的位置位于固定的地址偏移处。用户将第一个实例的A，B和C矩阵的指针与确定未来实例中输入和输出矩阵位置的地址偏移量strideA，strideB和strideC一起传递给函数。
2. 该函数的优化本质是矩阵乘法的优化。
3. 优化思想：
  + share memory
  + stream
  + Hierarchical Structure(硬件映射)(参考cutlass)
