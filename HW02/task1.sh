#!/usr/bin/env zsh
#SBATCH --job-name=task6
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --array=10-30
#SBATCH --afternotok:job_id[10:30] srun task1plotter.py

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# REF: https://zsh.sourceforge.io/Doc/Release/Arithmetic-Evaluation.html
let "n = 2 ** $SLURM_ARRAY_TASK_ID "

# REF: https://stackoverflow.com/questions/6022384/bash-tool-to-get-nth-line-from-a-file
# REF: https://man7.org/linux/man-pages/man1/flock.1.html
# REF: https://stackoverflow.com/questions/24388009/linux-flock-how-to-just-lock-a-file


CSV_STRING=$(./task1 $((n)) | sed '3q;d' | xargs printf "%s,%d\n" $SLURM_ARRAY_TASK_ID)

exec {lock_fd}>>runtime.csv
flock "$lock_fd"
echo "$CSV_STRING" >> runtime.csv
exec {lock_fd}>&-