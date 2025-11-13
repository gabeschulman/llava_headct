#!/bin/bash
#SBATCH --job-name=qa_parquet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/qa_parquet_%j.out
#SBATCH --error=logs/qa_parquet_%j.err

source ~/.bashrc
conda activate llava2
export PYTHONPATH="/gpfs/scratch/$USER:$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct/src/

python merge_qa_to_parquet.py
