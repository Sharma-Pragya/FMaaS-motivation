#!/bin/bash
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH --gres=gpu:a16:1 # Number and type of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-a100.out  # %j = job ID

module load conda/latest
module load cuda/12.6
conda activate benchmark-foundation-vqa
pip install flash-attn