#!/bin/bash

#SBATCH -J sgemm_batched
#SBATCH -p caspra
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib

srun ./example_sgemm_strided_batched
