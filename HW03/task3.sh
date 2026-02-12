#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:01:30
#SBATCH --gres=gpu:1 -c 1
#SBATCH --output=Slurm.out
#SBATCH --error=Slurm.err

module load nvidia/cuda/13.0.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# REF: https://zsh.sourceforge.io/Doc/Release/Arithmetic-Evaluation.html
# REF: https://stackoverflow.com/questions/6022384/bash-tool-to-get-nth-line-from-a-file
# REF: https://man7.org/linux/man-pages/man1/flock.1.html
# REF: https://stackoverflow.com/questions/24388009/linux-flock-how-to-just-lock-a-file

for idx in {9..29}; do
        let "n = 2 ** $idx"
        ./task3 $((n)) | sed '1q;d' | xargs printf "%d,%s\n" "$idx" >> runtime.csv
done

