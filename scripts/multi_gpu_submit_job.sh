#!/bin/bash
#SBATCH --job-name=llava_head_ct_pretrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:2
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
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

# Launch with torchrun for distributed training
torchrun --nnodes 1 --nproc_per_node 4 --master_port 12400 \
    src/trainer.py --config_name narrative_train_config

echo "Job completed at: $(date)"
