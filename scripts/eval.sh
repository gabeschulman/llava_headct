#!/bin/bash
#SBATCH --job-name=HeadCT_Eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --partition=a100_long
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source activate /gpfs/scratch/$USER/conda_envs/llava2

python narrative_eval.py
