#!/bin/bash
#SBATCH --job-name jpeg_metrics
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node v100:0
#SBATCH --mem 128gb
#SBATCH --time 7:00:00
#SBATCH --constraint interconnect_hdr


cd path_here_to_jpeg_metrics.py

source activate your_conda_environment

python jpeg_metrics.py
