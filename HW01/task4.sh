#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

# N.B.: Slide 16 of the Euler slide is misleading.
# It says that --cpus-per-task is for CPU threads.
# https://slurm.schedmd.com/cpu_management.html says its for the number of CPUs
# It can't be both, please so refer to the URL for resource allocation setup

hostname