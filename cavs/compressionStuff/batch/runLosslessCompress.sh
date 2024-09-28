#!/bin/bash
#SBATCH --job-name lossless_compress
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node v100:0
#SBATCH --mem 128gb
#SBATCH --time 7:00:00
#SBATCH --constraint interconnect_hdr

cd
source ./spack/share/spack/setup-env.sh

spack env activate yourSpackEnvNameHere

cd your_path_tolossless_compress.py_here

source yourPythonEnvNameHere/bin/activate

python lossless_compress.py $1
