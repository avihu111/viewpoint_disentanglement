#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python train_vae.py --kl_loss_weight 0.01
