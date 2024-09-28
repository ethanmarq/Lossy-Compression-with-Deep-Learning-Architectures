#!/bin/bash

#SBATCH --job-name train_UNETwRESNET
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 16gb
#SBATCH --time 12:00:00

source activate conda_env_name_here

cd path_name_here

python train.py "$1" "$2"
