#!/bin/bash
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH --gres=gpu:a100:1 # Number and type of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-a100.out  # %j = job ID

module load conda/latest
conda activate foundation-timeseries

python -m xiuhmolpilli.arena &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_tiny_a100.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID