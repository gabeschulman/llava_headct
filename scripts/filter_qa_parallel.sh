#!/bin/bash
#SBATCH --job-name=filter_q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=40:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_long
#SBATCH --output=logs/filter_q_%j.out
#SBATCH --error=logs/filter_q_%j.err

mkdir -p logs

source ~/.bashrc
conda activate head_ct
export PYTHONPATH="/gpfs/scratch/$USER:$PYTHONPATH"
cd /gpfs/scratch/$USER

export HF_HOME="/gpfs/scratch/$USER/.cache/huggingface"
mkdir -p $HF_HOME

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run with parallel processing
# Syntax: python filter_qa_parallel.py <input> [output] [batch_size]
# python filter_qa_parallel.py radiology_qa_results_checkpoint_part1.json radiology_qa_final_part1.json 32
python filter_qa_parallel.py radiology_qa_results_part3.json radiology_qa_final_part3.json 32

echo "End time: $(date)"
echo "Q filter complete!"