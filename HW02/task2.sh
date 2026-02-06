#!/usr/bin/env zsh
#SBATCH --job-name=taskq
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=Slurm.out
#SBATCH --error=Slurm.err

# REF: https://zsh.sourceforge.io/Doc/Release/Arithmetic-Evaluation.html
# REF: https://stackoverflow.com/questions/6022384/bash-tool-to-get-nth-line-from-a-file
# REF: https://man7.org/linux/man-pages/man1/flock.1.html
# REF: https://stackoverflow.com/questions/24388009/linux-flock-how-to-just-lock-a-file

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

./task2 5 5
