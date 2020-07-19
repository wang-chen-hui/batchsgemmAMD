#!/bin/bash	

#SBATCH -J test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -p caspra
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH --gres=dcu:4

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
./example_sgemm_strided_batched -m 512 -n 512 -k 256 --batch_count 10
