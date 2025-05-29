#!/bin/bash
#SBATCH --mem=24G  # Requested Memory
#SBATCH -p gpu # Partition
#SBATCH --gres=gpu:a100:1 # Number and type of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-a100.out  # %j = job ID

module load conda/latest
conda activate moirai
python -m src.main --mode fcst_moirai > log_moirai.txt 2>&1