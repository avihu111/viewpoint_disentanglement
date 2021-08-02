#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python train_lord_dalle.py --dalle_output_file_name dalle_realestate_28Jun_reg0.01 --dalle_path dalle_27Jun_big_high_reg.pt --lr_decay --taming --learning_rate 0.0001 --regularization 0.01