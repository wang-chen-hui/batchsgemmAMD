#INC_PATH = -I. -I/opt/rocm/hip/include -I./include
INC_PATH = -I. -I./include
LIB_PATH = lib/libcheckresult.so lib/libsgemm_strided_batched.so -lcrypto
#LIB_PATH = -L./lib -lcheckresult -lcrypto
ADD_FLAG = -w -lcudart	-O3

all: example_sgemm_strided_batched.o
		nvcc $^ ${INC_PATH} ${ADD_FLAG} ${LIB_PATH} -o example_sgemm_strided_batched
example_sgemm_strided_batched.o: example_sgemm_strided_batched.cu ./include/*.h
		nvcc -c $< ${ADD_FLAG} -o $@

clean:
		rm -rf *.o
		rm -rf example_sgemm_strided_batched
