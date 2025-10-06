#!/bin/bash
#SBATCH --job-name=toolrl_inference       # Job name
#SBATCH --output=toolrl_inference_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=toolrl_inference_%j.err         # Error log
#SBATCH --time=00:05:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luozheng@usc.edu
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=16G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
module load python/3.12.1

# Go to your project directory
cd /u/zluo8/sim2real/ToolRL/zheng

source ~/.bashrc
source /u/zluo8/sim2real/ToolRL/.venv/bin/activate

# Convert the notebook to a .py script
jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script
python test_toolrl.py
