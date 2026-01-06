USE_CUDA = True

from dataclasses import dataclass
import os
from pathlib import Path
from permutations import create_random_permutation_with_fixed_points
import shutil
from tokenization import NllbTokenizer, HuggingfaceTokenizer


@dataclass
class FinetuningParameters:
    base_model: str
    should_finetune: bool
    report_every: int
    validate_every: int
    patience: int
    batch_size: int
    num_training_steps: int
    freeze_encoder: bool
    freeze_decoder: bool
    gradient_accumulation_steps: int
    max_grad_norm: float
    dev_batches: int


def read_finetuning_params(config):
    """Reads the finetuning parameters into a dataclass."""
    params = config["finetuning_parameters"]
    f_params = FinetuningParameters(
        base_model=params["base_model"],
        should_finetune=params.get("finetune", True),
        report_every=params.get("report_every", 500),
        validate_every=params.get("validate_every", 500),
        patience=params.get("patience", 1000000000),
        batch_size=params["batch_size"],
        num_training_steps=params["num_steps"],
        freeze_decoder=params.get("freeze_decoder", False),
        freeze_encoder=params.get("freeze_encoder", False),
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
        max_grad_norm=params.get("max_grad_norm", 1.0),
        dev_batches=params.get("dev_batches", 100),
    )
    return f_params


def create_experiment_dir(config, config_file):
    """Creates a new experiment directory and copies the config file into it."""
    base_dir = config["model_dir"]
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)
    shutil.copy(config_file, Path(model_dir) / "experiment.json")
    return model_dir


def harvest_language_codes(config):
    """Creates a dictionary that maps (corpus, lang) pairs to language codes."""
    lang_codes = dict()
    for corpus in config["corpora"]:
        for key in config["corpora"][corpus]:
            lang_codes[(corpus, key)] = config["corpora"][corpus][key]["lang_code"]
    return lang_codes


def initialize_tokenizer(config):
    # TODO: generalize to separate src/tgt tokenizers
    params = config["finetuning_parameters"]
    model_name = params["base_model"]
    if model_name == "facebook/nllb-200-distilled-600M":
        tokenizer = NllbTokenizer("600M", max_length=128)  # set max length?
    elif model_name == "facebook/nllb-200-distilled-1.3B":
        tokenizer = NllbTokenizer("1.3B", max_length=128)
    else:
        tokenizer = HuggingfaceTokenizer(model_name, max_length=128)
    return tokenizer


def create_permutations(config, tokenizer):
    all_corpora = config["corpora"]
    permutations = dict()
    pmap = dict()
    for corpus in all_corpora:
        for language in all_corpora[corpus]:
            permutation_index = all_corpora[corpus][language]["permutation"]
            if permutation_index > 0:
                if permutation_index not in permutations:
                    permutations[permutation_index] = (
                        create_random_permutation_with_fixed_points(
                            len(tokenizer),
                            list(tokenizer.get_special_tokens().values()),
                        )
                    )
                pmap[(corpus, language)] = permutations[permutation_index]
    # save_permutation_map(pmap, Path(model_dir) / "permutations.json")
    return pmap
