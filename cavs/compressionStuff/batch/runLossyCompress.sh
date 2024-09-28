#!/bin/bash
#SBATCH --job-name lossy_compress
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node v100:0
#SBATCH --mem 128gb
#SBATCH --time 7:00:00
#SBATCH --constraint interconnect_hdr

cd
source ./spack/share/spack/setup-env.sh

spack env activate yourSpackEnvNameHere

cd your_path_to_lossy_compress.py_here

source your_python_env_name_here/bin/activate

python lossy_compress.py $1
