import argparse
from configure import create_experiment_dir
from configure import create_permutations
from configure import harvest_language_codes
from configure import initialize_tokenizers
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
from tokenization import CharacterTokenizer
import torch
from tqdm import tqdm
from transformers import Adafactor
from transformers import get_constant_schedule_with_warmup
from validate import evaluate_experiment

matplotlib.use("Agg")


# makes dev and training graph
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

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if ft_params.should_finetune:
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )

        num_warmup_steps = int(0.05 * ft_params.num_training_steps)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    else:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            lr=None,
            clip_threshold=1.0,
            weight_decay=0.01,
        )
        scheduler = None

    accumulation_steps = ft_params.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)

    def evaluate(model, dev_data, batches):
        model.eval()
        try:
            losses = []
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for _ in range(batches):
                    x, y, _, _ = dev_data.next_batch()
                    x = x.to(model.device)
                    y = y.to(model.device)
                    loss = model(**x, labels=y.input_ids).loss
                    losses.append(loss.item())
        finally:
            model.train()
        return float(np.mean(losses))

    cleanup()
    train_losses = []
    train_plot_x, train_plot_y = [], []
    dev_plot_x, dev_plot_y = [], []

    best_dev_loss = None
    steps_since_best = 0

    model.train()

    for step in tqdm(range(1, ft_params.num_training_steps + 1)):
        try:
            x, y, _, _ = train_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**x, labels=y.input_ids).loss
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * accumulation_steps)
            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ft_params.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger("GPU OOM during training step. Skipping batch.", to_stderr=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                continue
            else:
                raise

        if step % ft_params.report_every == 0:  # logging
            avg_train_loss = np.mean(train_losses[-ft_params.report_every :])
            logger(f"Step {step} (train): {avg_train_loss:.4f}")
            train_plot_x.append(step)
            train_plot_y.append(avg_train_loss)

        if step % ft_params.validate_every == 0:  # validation
            logger("Validating...")
            dev_loss = evaluate(model, dev_data, batches=ft_params.dev_batches)
            logger(f"Dev loss: {dev_loss:.4f}")

            dev_plot_x.append(step)
            dev_plot_y.append(dev_loss)

            # graph loss
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )

            # achieved new lower dev loss -> save best model
            if best_dev_loss is None or dev_loss < best_dev_loss:
                logger("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                logger(
                    f"No improvement. Patience: "
                    f"{ft_params.patience - steps_since_best}"
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
    src_tokenizer, tgt_tokenizer = initialize_tokenizers(ft_params)

    pmap = create_permutations(config, src_tokenizer)
    save_permutation_map(pmap, Path(experiment_dir) / "permutations.json")
    train_data = MixtureOfBitexts.create_from_config(
        config, "train", only_once_thru=False
    )
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=False)
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data,
        src_tokenizer,
        tgt_tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        src_tokenizer,
        tgt_tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
    )
    model = prepare_model_for_finetuning(ft_params)
    # resize embeddings matrix (add embeddings from new lang codes)
    model.resize_token_embeddings(len(src_tokenizer) + len(tgt_tokenizer))
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
