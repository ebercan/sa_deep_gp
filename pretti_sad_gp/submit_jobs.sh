#!/bin/bash
#SBATCH -J bo_poly_10reps
#SBATCH --array=1-10%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --partition short
#SBATCH -o logs/worker_%A_%a.out
#SBATCH -e logs/worker_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=enisbaris@mpi-magdeburg.mpg.de

mkdir -p logs
mkdir -p data/results
mkdir -p data/queue_state

source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

sleep $((RANDOM % 10))

python hpc_manager.py worker \
    --job_file slurm_jobs.json \
    --queue_dir data/queue_state \
    --save_dir data/results \
    --hardware CPU
