#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=webnlg_script_test
#SBATCH --mail-type=NONE
#SBATCH --mail-user=cheung.179@osu.edu
#SBATCH --partition white-1gpu

BASE_DIR=~/LM
cd $BASE_DIR

python GPT2_PPL_test2.py
# python GPT2_PPL_test.py
# python GPT2_PPL_ex.py
# python GPT2_word_surprisal.py
