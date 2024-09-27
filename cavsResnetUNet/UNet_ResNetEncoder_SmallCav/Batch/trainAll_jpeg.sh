#!/bin/bash

for i in `seq 0 20 100`
do
sbatch trainUNetJpeg.sh "$i"
done