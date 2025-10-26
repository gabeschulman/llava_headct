# LLaVA Head CT

Data directory: `/gpfs/data/razavianlab/data/capstone_f25/cached_images/`

Contains train/val/test splits.

## Bash config

Only run once (to prevent caches being built in your home directory):

```bash
echo "export HF_HOME=/gpfs/scratch/\$USER/hf_cache" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/gpfs/scratch/\$USER/hf_cache" >> ~/.bashrc
echo "export PIP_CACHE_DIR=/gpfs/scratch/\$USER/pip_cache" >> ~/.bashrc
echo "export PRE_COMMIT_HOME=/gpfs/scratch/\$USER/pre-commit_cache" >> ~/.bashrc
source ~/.bashrc
```

## Environment Setup
```bash
bash scripts/setup_env.sh
conda activate /gpfs/scratch/$USER/conda_envs/llava2
pre-commit install
```

To submit jobs run (e.g.)
```bash
sbatch scripts/submit_job_classsification.sh
```

^This prints the job ID from SLURM. To monitor the job, run:

```bash
tail -f logs/slurm_$JOB_ID.out
```

To view the error (if the job fails), run:

```bash
cat logs/slurm_$JOB_ID.err
```
