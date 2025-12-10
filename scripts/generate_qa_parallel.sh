#!/bin/bash
#SBATCH --job-name=generate_qa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=40:00:00
#SBATCH --mem=64G
#SBATCH --partition=a100_long
#SBATCH --output=logs/generate_qa_%j.out
#SBATCH --error=logs/generate_qa_%j.err

mkdir -p logs

source ~/.bashrc
conda activate head_ct
export PYTHONPATH="/gpfs/scratch/$USER:$PYTHONPATH"
cd /gpfs/scratch/$USER

export HF_HOME="/gpfs/scratch/$USER/.cache/huggingface"
mkdir -p $HF_HOME

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Test run: 200 records
# get row number: grep -n "29863376" /gpfs/data/razavianlab/capstone/2025_stroke/ct_accession_report_2506_deid.csv
# python generate_qa6.py -n 200 -c 100 -b 8
# python generate_qa6.py -s 218017 -c 1000 -b 32 -o radiology_qa_results_part2.json
python generate_qa.py -s 438050 -c 1000 -b 32 -o radiology_qa_results_part3.json

echo "End time: $(date)"
echo "Q&A generation complete!"