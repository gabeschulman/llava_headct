#!/bin/bash

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_YML="$SCRIPT_DIR/../environment.yml"

CONDA_ENV_PATH="/gpfs/scratch/$USER/conda_envs/llava2"

echo "Creating conda environment at: $CONDA_ENV_PATH"
echo "This avoids filling up your home directory quota."
echo ""

# Create the directory if it doesn't exist
mkdir -p "/gpfs/scratch/$USER/conda_envs"

# Create the environment
conda env create -f "$ENV_YML" -p "$CONDA_ENV_PATH"

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, use:"
echo "  conda activate $CONDA_ENV_PATH"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  alias activate_llava='conda activate $CONDA_ENV_PATH'"
echo ""
