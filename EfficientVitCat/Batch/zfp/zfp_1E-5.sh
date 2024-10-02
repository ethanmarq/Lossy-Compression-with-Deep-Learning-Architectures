#!/bin/bash

#SBATCH --job-name train_efficientVit
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 12gb
#SBATCH --time 08:00:00


module load anaconda3
source activate torch


cd /home/marque6/MLBD/LossyUGVPaper/EfficientVitCat


python train_seg_model.py configs/seg/cav/b0-lr3-n2-wd1.yaml --path runtest/zfp/zfp_1E5 --weight_url ckpts/b0_modified.pt --compress zfp --error 1E-5
