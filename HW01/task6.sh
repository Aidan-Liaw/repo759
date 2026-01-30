#!/usr/bin/env zsh
#SBATCH --job-name=task6
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err



hostname