#!/bin/bash
#SBATCH --job-name=com6912
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=70G
#SBATCH --output=./Output/out.txt
#SBATCH --mail-user=sdevanand1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,FAIL,END

module load Java/11.0.18
module load Anaconda3/2022.05
module load CUDA/11.8.0

export LANG=en_US.utf8
export LC_ALL=en_US.utf8


source activate myspark

# Check if 'transformers' is installed
if ! python -c "import transformers" &> /dev/null; then
    echo "Installing transformers..."
    pip install transformers
else
    echo "transformers is already installed"
fi

# Check if 'torch' is installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing torch..."
    pip install torch
else
    echo "torch  is already installed"
fi

pip install librosa
pip install soundfile
pip install accelerate -U


python script2.py
