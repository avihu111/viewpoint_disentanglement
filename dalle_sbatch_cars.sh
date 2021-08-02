#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python train_lord_dalle.py --dalle_path dalle_cars_Jun29_sigma1_reg_0.01_big.pt --lr_decay --taming --dataset cars --sigma 0.1 --regularization 0.01 --dalle_output_file_name dalle_cars_Jun29_sigma1_reg_0.01_big_sig0.1