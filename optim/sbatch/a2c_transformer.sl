#!/usr/bin/env bash
#SBATCH --job-name=a2c_transformer_rho_optim
#SBATCH --partition=epyc-gpu
#SBATCH --time=60:00:00

#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1

#SBATCH --output=/hpc/gpfs2/home/u/kueblero/embodied-scene-graph-navigation/optim/sbatch/logs/%x_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load anaconda
conda activate embodied-scene-graph-navigation

srun python ~/embodied-scene-graph-navigation/optim/rho_optimizer.py  --agent="a2c" --model="transformer"
