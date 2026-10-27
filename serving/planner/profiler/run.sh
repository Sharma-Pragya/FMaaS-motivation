#!/bin/bash
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gpu,gpu-preempt # Partition
#SBATCH --gres=gpu:l40s:1 # Number and type of GPUs (vLLM rows need the L40S)
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o ./logs/slurm-l40s.out  # %j = job ID

module load conda/latest
conda activate fmtk
python main.py combined_metrics_l40s.csv