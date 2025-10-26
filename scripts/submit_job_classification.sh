#!/bin/bash
#SBATCH --job-name=llava_head_ct_pretrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=50G
##SBATCH --mem=512G
#SBATCH --partition=radiology
#SBATCH --output=../log/llava_head_ct_pretrain__out.out
#SBATCH --error=../log/llava_head_ct_pretrain__err.err

mkdir -p /gpfs/scratch/gs4342/llava_headct/logs
mkdir -p /gpfs/scratch/gs4342/llava_headct/checkpoints
mkdir -p /gpfs/scratch/gs4342/llava_headct/models

# activate conda env
source activate /gpfs/scratch/gs4342/conda_envs/llava2

# module load cuda/12.1
# module load gcc/10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH="/gpfs/scratch/gs4342/llava_headct:$PYTHONPATH"
cd /gpfs/scratch/gs4342/llava_headct

# Log job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

torchrun --nnodes 1 --nproc_per_node 4 --master_port 12400 src/trainer.py --local_rank 0 \
    --model_name "llava_headct"

echo "Job completed at: $(date)"
