#!/bin/bash	

#SBATCH -J test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -p caspra
#SBATCH -o %j.log
#SBATCH -e %j.err
./example_sgemm_strided_batched
