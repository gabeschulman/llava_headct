#!/bin/bash
#SBATCH --job-name=preproc_val
#SBATCH --output=logs/preproc_val_%A_%a.out
#SBATCH --error=logs/preproc_val_%A_%a.err
#SBATCH --array=0-4  # 5 chunks for validation
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=cpu_short

source ~/.bashrc
source activate /gpfs/scratch/$USER/conda_envs/llava2
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

VAL_INPUT="/gpfs/scratch/$USER/ct_datasets_backup/nyu/nyu_val_processed.parquet"
CACHE_DIR="/gpfs/data/razavianlab/data/capstone_f25/cached_images"
TOTAL_CHUNKS=5

echo "Processing validation set chunk ${SLURM_ARRAY_TASK_ID}/${TOTAL_CHUNKS}..."
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

python src/preprocessing/cache_preprocessed_images_chunked.py \
    --input "$VAL_INPUT" \
    --output "$CACHE_DIR/val" \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --total-chunks ${TOTAL_CHUNKS} \
    --num-workers 10

echo "Chunk ${SLURM_ARRAY_TASK_ID} complete!"
