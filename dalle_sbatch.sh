#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python train_lord_dalle.py --vae_path ./vae_1510_new.pt --dalle_output_file_name dalle_small_1900 --lr_decay