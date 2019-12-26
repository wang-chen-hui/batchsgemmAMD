#!/bin/bash	

#SBATCH -J test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -p caspra
#SBATCH -o %j.log
#SBATCH -e %j.err
<<<<<<< HEAD

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
=======
./example_sgemm_strided_batched -m 1024 -n 512 -k 256 --batch_count 10
>>>>>>> bca2b99d49c529a2d391f631b3f92cd0b59e6a7a
