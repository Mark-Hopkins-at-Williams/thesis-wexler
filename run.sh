#!/bin/sh
#SBATCH -c 1
#SBATCH -t 4-22:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1


python finetune.py --config /mnt/storage/swexler/thesis-wexler/configs/french-training.json
# python scripts/organize_into_batches.py --in_dir examples/french-model_10_28_25/fr-en --out_dir examples/french-data-7-mil-512-filtered --tokenizer byte --max_len 512
# python autocomplete.py