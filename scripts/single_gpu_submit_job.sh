#!/bin/bash
#SBATCH --job-name=llava_head_ct_pretrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_short
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

mkdir -p /gpfs/scratch/$USER/llava_headct/logs
mkdir -p /gpfs/scratch/$USER/llava_headct/checkpoints
mkdir -p /gpfs/scratch/$USER/llava_headct/models

module load cuda/11.8

# activate conda env
source activate /gpfs/scratch/$USER/conda_envs/llava2

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to repo directory
cd /gpfs/scratch/$USER/llava_headct
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:$PYTHONPATH"

# Log job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

python src/trainer.py --job_id $SLURM_JOB_ID --config_name fine_tune_config

echo "Job completed at: $(date)"
