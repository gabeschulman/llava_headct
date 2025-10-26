#!/bin/bash
#SBATCH --job-name=preproc_train
#SBATCH --output=logs/preproc_train_%A_%a.out
#SBATCH --error=logs/preproc_train_%A_%a.err
#SBATCH --array=0-49%10  # 50 jobs, max 10 running simultaneously
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=cpu_short

# This processes training data in 50 chunks, but only runs 10 jobs at a time
# The %10 limits concurrent jobs to avoid hitting memory quota

source ~/.bashrc
source activate /gpfs/scratch/$USER/conda_envs/llava2
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

TRAIN_INPUT="/gpfs/scratch/$USER/ct_datasets_backup/nyu/nyu_train_processed.parquet"
CACHE_DIR="/gpfs/data/razavianlab/data/capstone_f25/cached_images"
TOTAL_CHUNKS=50

echo "Processing training set chunk ${SLURM_ARRAY_TASK_ID}/${TOTAL_CHUNKS}..."
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

python src/preprocessing/cache_preprocessed_images_chunked.py \
    --input "$TRAIN_INPUT" \
    --output "$CACHE_DIR/train" \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --total-chunks ${TOTAL_CHUNKS} \
    --num-workers 10

echo "Chunk ${SLURM_ARRAY_TASK_ID} complete!"
