USE_CUDA = True

from dataclasses import dataclass
import os
from pathlib import Path
from permutations import create_random_permutation_with_fixed_points
import shutil
from tokenization import NllbTokenizer, HuggingfaceTokenizer, CharacterTokenizer, ByteTokenizer, SentinelTokenizer


@dataclass
class FinetuningParameters:
    base_model: str
    src_tokenizer: str
    tgt_tokenizer: str
    max_src_length: int
    max_tgt_length: int
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
        src_tokenizer=params.get("src_tokenizer", "default"),
        tgt_tokenizer=params.get("tgt_tokenizer", "default"),
        max_src_length=params.get("max_src_length", 128),
        max_tgt_length=params.get("max_tgt_length", 128),
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


# initialize the tokenizers by creating each unique tokenizer necessary 
# then assinging them to src and tgt according to config
def initialize_tokenizers(ft_params): 
    tokenizer_types = set( # set gets rid of duplicates [ex: default, default]
        [
            (ft_params.src_tokenizer, ft_params.max_src_length),
            (ft_params.tgt_tokenizer, ft_params.max_tgt_length),
        ]
    )
    offset = 0
    tokenizers = dict()
    for tokenizer_type, max_length in tokenizer_types:
        if tokenizer_type == "default":
            model_name = ft_params.base_model
            if model_name == "facebook/nllb-200-distilled-600M":
                tokenizers[(tokenizer_type, max_length)] = NllbTokenizer("600M", max_length=max_length)
            elif model_name == "facebook/nllb-200-distilled-1.3B":
                tokenizers[(tokenizer_type, max_length)] = NllbTokenizer("1.3B", max_length=max_length)
            else:
                tokenizers[(tokenizer_type, max_length)] = HuggingfaceTokenizer(model_name, max_length=max_length)
            offset = len(tokenizers[(tokenizer_type, max_length)])
    for tokenizer_type, max_length in tokenizer_types: # this must be its own loop bc we need offset
        if tokenizer_type == "character":
            tokenizers[(tokenizer_type, max_length)] = CharacterTokenizer(max_length=max_length, offset=offset)
        if tokenizer_type == "byte":
            tokenizers[(tokenizer_type, max_length)] = ByteTokenizer(max_length=max_length, offset=offset)
        if tokenizer_type == "sentinel":
            tokenizers[(tokenizer_type, max_length)] = SentinelTokenizer(max_length=max_length, offset=offset)
    src_tokenizer = tokenizers[(ft_params.src_tokenizer, ft_params.max_src_length)]
    tgt_tokenizer = tokenizers[(ft_params.tgt_tokenizer, ft_params.max_tgt_length)]
    return src_tokenizer, tgt_tokenizer


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
