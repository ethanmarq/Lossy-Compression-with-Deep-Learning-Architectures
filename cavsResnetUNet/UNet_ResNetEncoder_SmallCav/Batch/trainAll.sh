#!/bin/bash

for i in `seq 1 7`
do
sbatch trainUNet.sh "$1" "$i"
done