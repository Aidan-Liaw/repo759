#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:03:00
#SBATCH --gres=gpu:1 -c 1
#SBATCH --output=Slurm.out
#SBATCH --error=Slurm.err

module load nvidia/cuda/13.0.0

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

# REF: https://zsh.sourceforge.io/Doc/Release/Arithmetic-Evaluation.html
# REF: https://stackoverflow.com/questions/6022384/bash-tool-to-get-nth-line-from-a-file
# REF: https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
# REF: https://stackoverflow.com/questions/4181703/how-to-concatenate-string-variables-in-bash

for idx in {9..29}; do
        let "n = 2 ** $idx"
        ./task2 $((n)) "$1" "$2" | sed '1q;d' | xargs printf "%d,%s\n" "$idx" >> "runtime${1}_${2}.csv"
done

