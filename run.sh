#!/bin/sh
#SBATCH -c 1
#SBATCH -t 4-22:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config /mnt/storage/swexler/thesis-wexler/configs/french-training.json
