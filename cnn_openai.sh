#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --exclude=gsm-04

dir=/cs/labs/daphna/avihu.dekel/DALLE-pytorch/
cd $dir
module spider cuda
module load cuda/11.1
source /cs/labs/daphna/avihu.dekel/stylegan2/style_venv/bin/activate

export PYTHONPATH=/cs/labs/daphna/avihu.dekel/DALLE-pytorch:/cs/labs/daphna/avihu.dekel/stylegan2

##python train_lord_dalle.py --dalle_output_file_name realestate_stylegan --lr_decay --dataset realestate --vae_type taming --generator stylegan --batch_size 192 --learning_rate 0.001 --epochs 1000 --dalle_path test.pt
.
python train_lord_dalle.py --dalle_output_file_name ffhq_transformer --lr_decay --dataset ffhq_small --vae_type openai --batch_size 20 --learning_rate 0.001


### srun --mem=10g -c 4 --time=1-0 --gres=gpu:1,vmem:20g--exclude=ampere-01 --pty $SHELL