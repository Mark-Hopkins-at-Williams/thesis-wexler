import argparse
from attention import SimpleAttention
from configure import create_experiment_dir
from configure import create_permutations
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
import json
import matplotlib
import matplotlib.pyplot as plt
from myutil import cleanup
from myutil import logger
from myutil import prepare_model_for_finetuning
import numpy as np
import os
from pathlib import Path
from permutations import save_permutation_map
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Adafactor
from transformers import get_constant_schedule_with_warmup
from validate import evaluate_experiment

matplotlib.use("Agg")


def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)


def finetune(model, train_data, dev_data, model_dir, ft_params):
    logger(f"Training {model_dir}")
    if ft_params.should_finetune:
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    else:  # use different optimizer for training from scratch
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            lr=None,  # required when using relative_step
            clip_threshold=1.0,
            weight_decay=0.01,
        )
        scheduler = None
    cleanup()
    train_losses, train_plot_x, train_plot_y = [], [], []
    dev_plot_x, dev_plot_y = [], []
    best_dev_loss, steps_since_best = None, 0
    encoder = model.model.encoder
    attn = SimpleAttention()
    for i in tqdm(range(ft_params.num_training_steps)):
        try:
            encoder.eval()
            src, tgt, _, _ = train_data.next_batch()
            src = src.to(encoder.device)
            tgt = tgt.to(encoder.device)
            src_enc = encoder(**src).last_hidden_state
            tgt_enc = encoder(**tgt).last_hidden_state
            out, _ = attn(src_enc, tgt_enc)
            token_scores = -torch.log(
                1 + F.cosine_similarity(src_enc, out, dim=-1) / 2
            )  # [seq_len]
            loss = token_scores.mean()
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            model.train()
            # x, y, _, _ = train_data.next_batch()
            # x = x.to(model.device)
            # y = y.to(model.device)
            loss = model(**src, labels=tgt.input_ids).loss
            loss.backward()
            # train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger("GPU OOM. Cleaning up.", to_stderr=True)
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e
        if i > 0 and i % ft_params.report_every == 0:
            avg_train_loss = np.mean(train_losses[-ft_params.report_every :])
            logger(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
        if i > 0 and i % ft_params.validate_every == 0:
            logger("Validating...")

            def evaluate(model, dev_data, batches: int = 100):
                model.eval()
                dev_losses = []
                with torch.no_grad():
                    for _ in range(batches):
                        x, y, _, _ = dev_data.next_batch()
                        x = x.to(model.device)
                        y = y.to(model.device)
                        loss = model(**x, labels=y.input_ids).loss
                        dev_losses.append(loss.item())
                return np.mean(dev_losses)

            dev_loss = evaluate(model, dev_data)
            logger(f"Dev loss: {dev_loss:.4f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss)
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )
            if best_dev_loss is None or dev_loss < best_dev_loss:
                logger("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                logger(
                    f"No improvement. Patience: {ft_params.patience - steps_since_best}"
                )
                if steps_since_best >= ft_params.patience:
                    logger("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()
    with open(args.config) as reader:
        config = json.load(reader)

    ft_params = read_finetuning_params(config)
    experiment_dir = create_experiment_dir(config, args.config)
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pmap = create_permutations(config, tokenizer)
    save_permutation_map(pmap, Path(experiment_dir) / "permutations.json")
    train_data = MixtureOfBitexts.create_from_config(
        config, "train", only_once_thru=False
    )
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=False)
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )
    model = prepare_model_for_finetuning(ft_params)
    finetune(
        model,
        tokenized_train,
        tokenized_dev,
        experiment_dir,
        ft_params,
    )
    evaluate_experiment(experiment_dir)


if __name__ == "__main__":
    main()
