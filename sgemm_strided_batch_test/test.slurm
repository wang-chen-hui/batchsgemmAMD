#!/bin/bash	

#SBATCH -J test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -p caspra
#SBATCH -o %j
#SBATCH -e %j
./example_sgemm_strided_batched > result
