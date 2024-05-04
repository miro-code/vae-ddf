#! /bin/bash
#SBATCH --nodes=1
#SBATCH --clusters=htc
#SBATCH --job-name=vae-for-ddf
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --account=engs-pnpl
#SBATCH --output=slurm_out/%j.out

module load Anaconda3/2022.10
conda activate /data/coml-oxmedis/trin4076/arc_nt_env

export WANDB_CACHE_DIR=$DATA/wandb_cache

python train_vae.py