#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:30
#SBATCH --gres=gpu:1 -c 1
#SBATCH --output=Slurm.out
#SBATCH --error=Slurm.err

module load nvidia/cuda/13.0.0

nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

./task2
