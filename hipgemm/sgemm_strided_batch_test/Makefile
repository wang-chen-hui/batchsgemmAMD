INC_PATH = -I. -I/opt/rocm/hip/include -I./include
LIB_PATH = -L./lib -lcheckresult -lsgemm_strided_batched -lcrypto
ADD_FLAG = -w

all: example_sgemm_strided_batched.o
		/opt/rocm/hip/bin/hipcc $^ ${INC_PATH} ${ADD_FLAG} ${LIB_PATH} -o example_sgemm_strided_batched
example_sgemm_strided_batched.o: example_sgemm_strided_batched.cpp
		/opt/rocm/hip/bin/hipcc -c $< ${INC_PATH} ${ADD_FLAG} ${LIB_PATH} -o $@

clean:
		rm -rf *.o
		rm -rf example_sgemm_strided_batched
