from transformers import AutoModelForSeq2SeqLM
import torch
import time
from tqdm import tqdm
from autocomplete import (
    tokenize,
    FileBasedLMData,
    DecoderOnlyTransformer,
    collate_causal_lm,
)
from torch.utils.data import DataLoader
from pathlib import Path
from corpora import Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from validate import translate_tokenized_mixture_of_bitexts, translate
from tokenization import (
    NllbTokenizer,
    HuggingfaceTokenizer,
    ByteTokenizer,
    SentinelTokenizer,
)
import json
from configure import (
    read_finetuning_params,
    harvest_language_codes,
    initialize_tokenizers,
    USE_CUDA,
)
import os
from myutil import logger
from file_examining import copy_first_lines


def get_full_translation_time(experiment_dir, compressed_test, num_lines=-1):
    # torch.cuda.set_device(0)
    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    model.cuda()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if __name__ == "__main__"
        else torch.device("cpu")
    )

    # LOAD AUTOCOMPLETE MODEL (BEFORE TIMER STARTS)
    autocomplete_model = DecoderOnlyTransformer(
        vocab_size=263,
        d_model=1024,
        nhead=16,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_len=1024,
    ).to(device)

    # LOAD CONFIG
    config_file = Path(experiment_dir) / "experiment.json"
    with open(config_file) as reader:
        config = json.load(reader)

    # GET EXPERIMENT NUMBER
    experiment_number = experiment_dir.split("-")[-1]
    experiment_number = experiment_number[:-1]

    # GET SUBSETS OF TEST DATA FOR QUICKER EXPERIMENTS
    if num_lines != -1:
        copy_first_lines(
            "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng",
            "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.eng",
            num_lines,
        )
        copy_first_lines(
            "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr",
            "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr",
            num_lines,
        )

    # COMPRESS TEST SET
    if compressed_test == True:
        start_compressed = time.time()
        if num_lines == -1:
            test_set_path = "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng"
        else:
            test_set_path = "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.eng"

        test_dataset = FileBasedLMData(test_set_path, max_length=1024)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
        )

        # Determine compression settings from the dataset used in experiment
        path = config["corpora"]["fr-eng"]["eng"]["train"]
        dirname = os.path.basename(os.path.dirname(path))
        parts = dirname.split("-")

        specifications = {
            "source_data": os.path.dirname(test_set_path),
            "compression_model": "/mnt/storage/swexler/thesis-wexler/models/autocomplete-v5",
            "prediction_threshold": float(parts[-3]),  # float within (0,1)
            "prediction_mode": parts[-2],  # margin or top_pred (absolute)
            "output_style": "short",  # short or long
            "date_compressed": "4_26_26",
            "autocomplete_mode": parts[-1],
        }
        desired_output_dir = (
            "/mnt/storage/swexler/thesis-wexler/examples/time-directory/"
        )

        tokenize(  # actually compresses
            autocomplete_model,
            test_loader,
            model_dir=specifications["compression_model"],
            output_dir=desired_output_dir,
            examples_type="test",
            prediction_threshold=specifications["prediction_threshold"],
            prediction_mode=specifications["prediction_mode"],
            output_style=specifications["output_style"],
            autocomplete_mode=specifications["autocomplete_mode"],
        )
        time_to_compress = time.time() - start_compressed
        logger(
            f"COMPRESSION TIME (exp dir {experiment_number}) ({'all' if num_lines == -1 else num_lines} lines) = {round(time_to_compress,3)}"
        )

    # PREP FOR TRANSLATION
    start_non_compressed = time.time()
    ft_params = read_finetuning_params(config)
    if num_lines == -1:
        if compressed_test == True:
            text_files = {
                (
                    "test",
                    "eng",
                ): f"{desired_output_dir}compressed-test-short.eng",  # use the compressed data we created above
                (
                    "test",
                    "fra",
                ): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr",
            }
        else:
            text_files = {
                (
                    "test",
                    "eng",
                ): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng",
                (
                    "test",
                    "fra",
                ): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr",
            }
    else:
        if compressed_test == True:
            text_files = {
                (
                    "test",
                    "eng",
                ): f"{desired_output_dir}compressed-test-short.eng",  # use the compressed data we created above
                (
                    "test",
                    "fra",
                ): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr",
            }
        else:
            text_files = {
                (
                    "test",
                    "eng",
                ): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.eng",
                (
                    "test",
                    "fra",
                ): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr",
            }

    ## TOKENIZE TEST DATASET
    lang_codes = harvest_language_codes(config)
    src_tokenizer, tgt_tokenizer = initialize_tokenizers(ft_params)

    lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
    test_data = MixtureOfBitexts.create_from_files(
        text_files,
        [(("test", "eng"), ("test", "fra"), None)],
        batch_size=32,
        only_once_thru=True,
    )
    logger("BEGINNING TOKENIZATION")
    # logger(src_tokenizer)
    # logger(tgt_tokenizer)
    # logger(test_data)
    # logger(lang_codes)
    tokenized_test = TokenizedMixtureOfBitexts(
        test_data,
        src_tokenizer,
        tgt_tokenizer,
        lang_codes=lang_codes,
    )

    ## TRANSLATE TEST DATASET
    logger("BEGINNING TRANSLATION")
    with torch.no_grad():
        batch = tokenized_test.next_batch()
        while batch is not None:
            x, y, _, _ = batch
            x = x.to(model.device)
            y = y.to(model.device)
            model(**x, labels=y.input_ids)
            batch = tokenized_test.next_batch()

    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tgt_tokenizer, lang_codes
    )

    # PRINT RESULTS
    if compressed_test == True:
        duration = time.time() - start_compressed
        logger(
            f"COMPRESSED DURATION (exp dir {experiment_number}) ({'all' if num_lines == -1 else num_lines} lines) = {round(duration, 3)}"
        )
    else:
        duration = time.time() - start_non_compressed
        logger(
            f"STANDARD DURATION = (exp dir {experiment_number}) ({'all' if num_lines == -1 else num_lines} lines) = {round(duration, 3)}"
        )
    print("------")


if __name__ == "__main__":
    for i in range(1):
        get_full_translation_time(
            "/mnt/storage/swexler/thesis-wexler/models/french-training-v126/",
            compressed_test=False,
        )  # Byte-BPE 605k steps
        get_full_translation_time(
            "/mnt/storage/swexler/thesis-wexler/models/french-training-v110/",
            compressed_test=True,
        )  # Compressed, 57.5%, stylized letters
        get_full_translation_time(
            "/mnt/storage/swexler/thesis-wexler/models/french-training-v109/",
            compressed_test=True,
        )  # compressed, 44.7%, stylized letters
        get_full_translation_time(
            "/mnt/storage/swexler/thesis-wexler/models/french-training-v117/",
            compressed_test=True,
        )  # Compressed, 32.1%, stylized letters
        get_full_translation_time(
            "/mnt/storage/swexler/thesis-wexler/models/french-training-v82/",
            compressed_test=False,
        )  # BPE-BPE 200k steps
        print()
        print()
