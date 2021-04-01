#!/bin/bash

#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o %j_stdout.txt
#SBATCH -e %j_stderr.txt
#SBATCH --gres=gpu

network="resnet34"
src="real"
tgt="clipart"
num_label=3

python s1_trainval_baseline.py --net $network --source $src --target $tgt --num $num_label
python s2_eval_and_save_features.py --net $network --source $src --target $tgt --num $num_label
python s3_selective_pseudo_labeling.py --net $network --source $src --target $tgt --num $num_label
python s4_trainval_prog_self_training.py --net $network --source $src --target $tgt --num $num_label
