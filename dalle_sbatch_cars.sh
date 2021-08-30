#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:10g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
## python train_lord_dalle.py --dalle_path dalle_cars_Jun29_sigma1_reg_0.01_big.pt --lr_decay --taming --dataset cars --sigma 0.1 --regularization 0.01 --dalle_output_file_name dalle_cars_Jun29_sigma1_reg_0.01_big_sig0.1
## --dalle_output_file_name test --lr_decay --taming --attn_types axial_row,axial_col --dataset cars


## python train_lord_dalle.py --dalle_output_file_name cars_09Aug_no_regularization --lr_decay --dataset cars --sigma 0 --regularization 0


python train_lord_dalle.py --dalle_output_file_name cnn_generator_cars_12Aug --lr_decay --dataset cars --cnn_generator --learning_rate 0.0001 --dalle_path cnn_generator_cars_cheating2.pt