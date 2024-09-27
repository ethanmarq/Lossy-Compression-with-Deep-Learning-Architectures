#!/bin/bash

#SBATCH --job-name train_validate_unet_lossy
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 128gb
#SBATCH --time 4:00:00

source activate universal

cd /home/aniemcz/cavsResnetUNet/UNet_ResNetEncoder_SmallCav/

python val.py "$1"
