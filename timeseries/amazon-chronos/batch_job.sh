#!/bin/bash
#SBATCH --mem=24G  # Requested Memory
#SBATCH -p gpu # Partition
#SBATCH --gres=gpu:a100:1 # Number and type of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-a100.out  # %j = job ID

module load conda/latest
conda activate amazon-chronos
python -m src.main --mode fcst_chronos > log_amazon.txt 2>&1