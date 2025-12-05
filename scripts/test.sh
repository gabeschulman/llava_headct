#!/bin/bash
#SBATCH --job-name=HeadCT_Eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00
#SBATCH --mem=10G
#SBATCH --partition=a100_dev
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

source activate /gpfs/scratch/$USER/conda_envs/llava2

python random_testing.py
