#!/bin/bash
# Submit multiple preprocessing jobs in parallel for chunked processing
# This allows processing 217k images much faster by using multiple nodes

NUM_CHUNKS=50  # Split into 50 chunks for maximum parallelization
CPUS_PER_JOB=32
MEM_PER_JOB=64GB
TIME_LIMIT=4:00:00

# Define paths
TRAIN_INPUT="/$USER/scratch/gs4342/ct_datasets_backup/nyu/nyu_train_processed.parquet"
VAL_INPUT="/$USER/scratch/gs4342/ct_datasets_backup/nyu/nyu_val_processed.parquet"
TEST_INPUT="/$USER/scratch/gs4342/ct_datasets_backup/nyu/nyu_test_processed.parquet"
CACHE_DIR="/$USER/scratch/gs4342/ct_datasets_backup/nyu/cached_images"

mkdir -p logs

echo "Submitting $NUM_CHUNKS parallel jobs for training set..."

# Submit jobs for training set
for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=preproc_train_${chunk_id}
#SBATCH --output=logs/preproc_train_${chunk_id}_%j.out
#SBATCH --error=logs/preproc_train_${chunk_id}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --partition=cpu_short

source ~/.bashrc
source activate /gpfs/scratch/$USER/conda_envs/llava2
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:\$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

echo "Processing training set chunk ${chunk_id}/${NUM_CHUNKS}..."
python src/preprocessing/cache_preprocessed_images_chunked.py \\
    --input "$TRAIN_INPUT" \\
    --output "$CACHE_DIR/train" \\
    --chunk-id ${chunk_id} \\
    --total-chunks ${NUM_CHUNKS} \\
    --num-workers $((CPUS_PER_JOB - 2))

echo "Chunk ${chunk_id} complete!"
EOF
done

echo "Submitted $NUM_CHUNKS jobs for training set!"

# Also submit validation and test sets (smaller, so fewer chunks)
NUM_VAL_CHUNKS=5
NUM_TEST_CHUNKS=5

echo ""
echo "Submitting $NUM_VAL_CHUNKS parallel jobs for validation set..."

for chunk_id in $(seq 0 $((NUM_VAL_CHUNKS-1))); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=preproc_val_${chunk_id}
#SBATCH --output=logs/preproc_val_${chunk_id}_%j.out
#SBATCH --error=logs/preproc_val_${chunk_id}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --partition=cpu_short

source ~/.bashrc
source activate /gpfs/scratch/$USER/conda_envs/llava2
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:\$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

echo "Processing validation set chunk ${chunk_id}/${NUM_VAL_CHUNKS}..."
python src/preprocessing/cache_preprocessed_images_chunked.py \\
    --input "$VAL_INPUT" \\
    --output "$CACHE_DIR/val" \\
    --chunk-id ${chunk_id} \\
    --total-chunks ${NUM_VAL_CHUNKS} \\
    --num-workers $((CPUS_PER_JOB - 2))

echo "Chunk ${chunk_id} complete!"
EOF
done

echo "Submitted $NUM_VAL_CHUNKS jobs for validation set!"
echo ""
echo "Submitting $NUM_TEST_CHUNKS parallel jobs for test set..."

for chunk_id in $(seq 0 $((NUM_TEST_CHUNKS-1))); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=preproc_test_${chunk_id}
#SBATCH --output=logs/preproc_test_${chunk_id}_%j.out
#SBATCH --error=logs/preproc_test_${chunk_id}_%j.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --partition=cpu_short

source ~/.bashrc
source activate /gpfs/scratch/$USER/conda_envs/llava2
export PYTHONPATH="/gpfs/scratch/$USER/llava_headct:\$PYTHONPATH"
cd /gpfs/scratch/$USER/llava_headct

echo "Processing test set chunk ${chunk_id}/${NUM_TEST_CHUNKS}..."
python src/preprocessing/cache_preprocessed_images_chunked.py \\
    --input "$TEST_INPUT" \\
    --output "$CACHE_DIR/test" \\
    --chunk-id ${chunk_id} \\
    --total-chunks ${NUM_TEST_CHUNKS} \\
    --num-workers $((CPUS_PER_JOB - 2))

echo "Chunk ${chunk_id} complete!"
EOF
done

echo "Submitted $NUM_TEST_CHUNKS jobs for test set!"
echo ""
echo "================================================"
echo "SUBMITTED $(($NUM_CHUNKS + $NUM_VAL_CHUNKS + $NUM_TEST_CHUNKS)) TOTAL JOBS!"
echo "================================================"
echo ""
echo "Train: $NUM_CHUNKS jobs (~4,356 images per chunk)"
echo "Val:   $NUM_VAL_CHUNKS jobs"
echo "Test:  $NUM_TEST_CHUNKS jobs"
echo ""
echo "To monitor progress:"
echo "  watch -n 5 'squeue -u \$USER | grep preproc'"
echo ""
echo "To check completion:"
echo "  squeue -u \$USER | grep preproc | wc -l"
echo ""
echo "After all jobs complete, merge chunks:"
echo ""
echo "  # Train set"
echo "  python src/preprocessing/merge_cached_chunks.py \\"
echo "    --input $TRAIN_INPUT \\"
echo "    --cache-dir $CACHE_DIR/train \\"
echo "    --total-chunks $NUM_CHUNKS"
echo ""
echo "  # Val set"
echo "  python src/preprocessing/merge_cached_chunks.py \\"
echo "    --input $VAL_INPUT \\"
echo "    --cache-dir $CACHE_DIR/val \\"
echo "    --total-chunks $NUM_VAL_CHUNKS"
echo ""
echo "  # Test set"
echo "  python src/preprocessing/merge_cached_chunks.py \\"
echo "    --input $TEST_INPUT \\"
echo "    --cache-dir $CACHE_DIR/test \\"
echo "    --total-chunks $NUM_TEST_CHUNKS"
echo ""
