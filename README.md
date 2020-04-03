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
