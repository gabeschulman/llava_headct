#!/bin/bash
#SBATCH --job-name=qa_parquet_combine
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/qa_parquet_%j.out
#SBATCH --error=logs/qa_parquet_%j.err

# Activate conda environment
source ~/.bashrc
conda activate llava2

# Set Python path
export PYTHONPATH="/gpfs/scratch/$USER:$PYTHONPATH"

# Change to working directory
cd /gpfs/scratch/$USER/llava_headct/src/

# Run the merge script
echo "Starting Q&A parquet merge..."
echo "Combining 3 Q&A JSON files and merging with cached parquet files"
echo "Output will be saved to: /gpfs/scratch/bm3772/data/all_questions"
echo ""

python merge_qa_to_parquet_combined.py

echo ""
echo "Job completed!"