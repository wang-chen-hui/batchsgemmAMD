#!/bin/bash	

#SBATCH -J test
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p caspra
#SBATCH -o %j
#SBATCH -e %j
./a.out > result
