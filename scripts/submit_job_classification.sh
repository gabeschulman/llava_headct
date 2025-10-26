#!/bin/bash
#SBATCH --job-name=llava_head_ct_pretrain_$USER
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=/gpfs/scratch/$USER/llava_headct/logs/slurm_%j.out
#SBATCH --error=/gpfs/scratch/$USER/llava_headct/logs/slurm_%j.err

mkdir -p /gpfs/scratch/$USER/llava_headct/logs
mkdir -p /gpfs/scratch/$USER/llava_headct/checkpoints
mkdir -p /gpfs/scratch/$USER/llava_headct/models

module load cuda/11.8

# activate conda env
source activate /gpfs/scratch/$USER/conda_envs/llava2

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

# Log job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

python src/trainer.py

echo "Job completed at: $(date)"
