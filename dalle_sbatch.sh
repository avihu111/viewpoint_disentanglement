#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:10g

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python train_lord_dalle.py --dalle_output_file_name re_08Aug_cnn --lr_decay --dalle_path dalle_realestate_28Jun.pt --cnn_generator