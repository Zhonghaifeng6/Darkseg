#!/bin/bash
#SBATCH --job-name=darkseg
#SBATCH --output=./logs/%j_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

NUM_GPUS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port 29567\
    train.py \
    --output_dir ./checkpoints/ \