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
echo "Creating conda environment (this may take a while)..."
conda env create -f "$ENV_YML" -p "$CONDA_ENV_PATH"

echo ""
echo "Activating environment to install PyTorch with CUDA support..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

echo ""
echo "Installing PyTorch with CUDA 11.8 support..."
pip install --upgrade torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, use:"
echo "  conda activate $CONDA_ENV_PATH"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  alias activate_llava='conda activate $CONDA_ENV_PATH'"
echo ""
