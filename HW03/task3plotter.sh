#!/usr/bin/env zsh
#SBATCH --job-name=taskq
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:05:00
#SBATCH --output=Slurm.out
#SBATCH --error=Slurm.err
#SBATCH --mem=0

python task3plotter.py
