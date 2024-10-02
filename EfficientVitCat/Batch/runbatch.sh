#!/bin/bash

#SBATCH --job-name train_efficientVit
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 12gb
#SBATCH --time 00:05:00



cd /home/marque6/MLBD/LossyUGVPaper/EfficientVitCat/Batch


# Submit all zfp jobs
cd ./zfp
for script in *.sh; do
    sbatch "$script"
done

# Submit all sz3 jobs
cd ../sz3
for script in *.sh; do
    sbatch "$script"
done

# Submit all jpeg jobs
cd ../jpeg
for script in *.sh; do
    sbatch "$script"
done

