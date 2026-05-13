from transformers import AutoModelForSeq2SeqLM
import torch
import time
from tqdm import tqdm
from autocomplete import tokenize, FileBasedLMData, DecoderOnlyTransformer, collate_causal_lm
from torch.utils.data import DataLoader
from pathlib import Path
from corpora import Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from validate import translate_tokenized_mixture_of_bitexts, translate
from tokenization import NllbTokenizer, HuggingfaceTokenizer, ByteTokenizer, SentinelTokenizer
import json
from configure import read_finetuning_params, harvest_language_codes, initialize_tokenizers, USE_CUDA
import os
from myutil import logger
from file_examining import copy_first_lines
import random
from collections import defaultdict
import statistics as stats


def get_full_translation_time(experiment_dir, compressed_test, num_lines=-1):
  # torch.cuda.set_device(0)
  # LOAD TRANSLATION MODEL
  model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
  model.cuda()

  # SET DEVICE
  device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if __name__ == "__main__"
    else torch.device("cpu")
  )

  # LOAD AUTOCOMPLETE MODEL
  autocomplete_model = DecoderOnlyTransformer(
      vocab_size=263,
      d_model=1024,
      nhead=16,
      num_layers=2,
      dim_feedforward=512,
      dropout=0.1,
      max_len=1024,
    ).to(device)

  # LOAD AND READ EXPERIMENT CONFIG
  config_file = Path(experiment_dir) / "experiment.json"
  with open(config_file) as reader:
      config = json.load(reader)
  ft_params = read_finetuning_params(config)
  lang_codes = harvest_language_codes(config)
  src_tokenizer, tgt_tokenizer = initialize_tokenizers(ft_params)

  # GET EXPERIMENT NUMBER
  experiment_number = experiment_dir.split("-")[-1]
  experiment_number = experiment_number[:-1]

  # GET SUBSETS OF TEST DATA FOR QUICKER EXPERIMENTS
  if num_lines != -1:
    copy_first_lines("/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng", 
                     "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.eng", 
                     num_lines)
    copy_first_lines("/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr", 
                     "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr", 
                     num_lines)

  # SET PATHS FOR ENG AND FR TEST SETS
  desired_output_dir = "/mnt/storage/swexler/thesis-wexler/examples/time-directory/" # for compression
  if compressed_test == False: # not compressing (byte / BPE)
    if num_lines == -1: # using all data
      text_files = {
        ("test", "eng"): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng",
        ("test", "fra"): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr",
      }
    else: # using subset of data
      text_files = {
        ("test", "eng"): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.eng",
        ("test", "fra"): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr",
      }
  if compressed_test == True: # compressed data
    if num_lines == -1: # using all data
      text_files = {
        ("test", "eng"): f"{desired_output_dir}compressed-test-short.eng", # use the compressed data we created above
        ("test", "fra"): "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.fr",
      }
    else: # using subset of data
      text_files = {
        ("test", "eng"): f"{desired_output_dir}compressed-test-short.eng", # use the compressed data we created above
        ("test", "fra"): "/mnt/storage/swexler/thesis-wexler/examples/time-directory/test_shorter.fr",
      }



  # COMPRESS TEST SET
  if compressed_test == True:
    # SETUP FOR COMPRESSION
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
        "autocomplete_mode": parts[-1]
    }

    torch.cuda.synchronize()
    time_after_setting_up_compressed = time.time() # =================TIMER STARTS===========================

    # ACTUALLY COMPRESS
    tokenize( 
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
    torch.cuda.synchronize()
    time_after_compressing = time.time()


  ## BEGINNING OF TRANSLATION PIPELINE
  time_after_setting_up_non_compressed = time.time() # =================TIMER STARTS===========================
  
  # SET UP REFERENCES FOR TOKENIZATION
  time_before_tokenizing_setup = time.time()

  lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
  test_data = MixtureOfBitexts.create_from_files(
      text_files,
      [(("test", "eng"), ("test", "fra"), None)],
      batch_size=1,
      only_once_thru=True,
  )

  tokenized_test = TokenizedMixtureOfBitexts(
      test_data,
      src_tokenizer,
      tgt_tokenizer,
      lang_codes=lang_codes,
  )
  time_after_tokenizing_setup = time.time()

  # TRANSLATE TEST DATASET
  logger("--beginning translation")
  with torch.no_grad():
    total_encoding_time = 0
    total_decoding_time = 0
    total_tokenization_time = 0
    batch = tokenized_test.next_batch()
    while batch is not None:
        x, y, _, _ = batch
        x = x.to(model.device)
        y = y.to(model.device)

        # encoding
        torch.cuda.synchronize()
        time_before_encoding = time.time()
        encoder_outputs = model.get_encoder()(**x)
        torch.cuda.synchronize()
        time_after_encoding = time.time()

        print(x["input_ids"].shape, x["attention_mask"].sum())

        # decoding
        model.generate(
          encoder_outputs=encoder_outputs,
          attention_mask=x["attention_mask"],
          forced_bos_token_id=tgt_tokenizer.get_special_tokens()["fra_Latn"],
          max_new_tokens=128,
          num_beams=1,
          do_sample=False,
        )
        torch.cuda.synchronize()
        time_after_decoding = time.time()

        # adding this batch's times to total time
        total_encoding_time += time_after_encoding - time_before_encoding
        total_decoding_time += time_after_decoding - time_after_encoding
        
        # getting next batch (which includes tokenization)
        time_before_tokenizing = time.time()
        batch = tokenized_test.next_batch()
        time_after_tokenizing = time.time()

        total_tokenization_time += time_after_tokenizing - time_before_tokenizing
  torch.cuda.synchronize()
  time_after_translating = time.time()


  # RETURN RESULTS
  if compressed_test == True:
    return {
      "experiment_type": src_tokenizer,
      "experiment_dir": experiment_number,
      "num_lines": 'all' if num_lines == -1 else num_lines,
      "compression_time": round(time_after_compressing - time_after_setting_up_compressed, 3),
      "tokenization_time": round(total_tokenization_time + time_after_tokenizing_setup - time_before_tokenizing_setup, 3),
      "encoding_time": round(total_encoding_time, 3),
      "decoding_time": round(total_decoding_time, 3),
      "translation_time": round(time_after_translating - time_after_compressing, 3),
      "end_to_end_time": round(time_after_translating - time_after_setting_up_compressed, 3)
    }

  else:
    return {
      "experiment_type": src_tokenizer,
      "experiment_dir": experiment_number,
      "num_lines": 'all' if num_lines == -1 else num_lines,
      "tokenization_time": round(total_tokenization_time + time_after_tokenizing_setup - time_before_tokenizing_setup, 3),
      "encoding_time": round(total_encoding_time, 3),
      "decoding_time": round(total_decoding_time, 3),
      "translation_time": round(time_after_translating - time_after_setting_up_non_compressed, 3),
      "end_to_end_time": round(time_after_translating - time_after_setting_up_non_compressed, 3)
    }




# DRIVER CODE
if __name__ == "__main__":

  # SPECIFICATIONS OF WHAT WE ARE TESTING
  num_lines = 640
  num_iterations = 1
  experiments = [
    ("/mnt/storage/swexler/thesis-wexler/models/french-training-v126/", False, num_lines, "Byte"),
    ("/mnt/storage/swexler/thesis-wexler/models/french-training-v110/", True, num_lines, "Compressed-57.5"),
    ("/mnt/storage/swexler/thesis-wexler/models/french-training-v109/", True, num_lines, "Compressed-44.7"),
    ("/mnt/storage/swexler/thesis-wexler/models/french-training-v117/", True, num_lines, "Compressed-32.1"),
    ("/mnt/storage/swexler/thesis-wexler/models/french-training-v140/", False, num_lines, "BPE"),
  ]

  # MAKING DATA STRUCTURES WHERE RESULTS WILL BE STORED
  results_compression = defaultdict(list) # storing compression time
  results_tokenization = defaultdict(list)
  results_translation = defaultdict(list) # storing translation only time
  results_encoding = defaultdict(list) # storing time to encode
  results_decoding = defaultdict(list) # storing time to decode
  results_end_to_end = defaultdict(list) # storing end to end time

  # RUNNING THE ABOVE METHOD
  for i in range(1, num_iterations+1):
    print(f"Iteration {i}")

    shuffled_experiments = experiments[:] # shallow copy of experiments
    random.shuffle(shuffled_experiments) # shuffle order

    # time each experiment as specified above
    for (exp_dir, compressed_test_local, num_lines_local, label) in shuffled_experiments:
      result = get_full_translation_time(exp_dir, compressed_test=compressed_test_local, num_lines=num_lines_local)
      print(label, result)
      
      # add time to that experiment's list of times
      results_end_to_end[label].append(result["end_to_end_time"])
      results_translation[label].append(result["translation_time"])
      results_encoding[label].append(result["encoding_time"])
      results_decoding[label].append(result["decoding_time"])
      results_tokenization[label].append(result["tokenization_time"])
      if compressed_test_local == True:
        results_compression[label].append(result["compression_time"])

    print()
    print()

  all_results = { # segments we are timing
    "end_to_end": results_end_to_end, # k = experiment type, v = list of times
    "translation": results_translation,
    "compression": results_compression,
    "tokenization": results_tokenization,
    "encoding": results_encoding,
    "decoding": results_decoding,
  }

  # get stats on a list of times
  def summarize(times):
    times = sorted(times)

    # the statistics library interpolates times and I want recorded times
    def nearest_percentile(times, p):
      idx = round(p * (len(times) - 1))
      return sorted(times)[idx]
    return {
        "mean": round(stats.mean(times), 3),
        "median": round(stats.median(times), 3),
        "sd": round(stats.stdev(times), 3) if len(times) > 1 else 0,
        "min": round(times[0], 3),
        "max": round(times[-1], 3),
        "p25": round(nearest_percentile(times, 0.25), 3),
        "p75": round(nearest_percentile(times, 0.75), 3),
  }

  # use summarize() to get stats on each segment in each experiment type
  # summary is formatted as {segment, {experiment, {statistic, value}}}
  summary = {
    segment: {k: summarize(v) for k, v in experiments.items()}
    for segment, experiments in all_results.items()
  }

  # ensure experiments are listed in the same order each time
  def sort_dict(d):
    return dict(sorted(d.items()))

  # print aggregates in readable format
  for metric, experiments in summary.items():
      print(f"\n{metric.upper()}")
      for exp, stats_dict in sort_dict(experiments).items():
          print(f"{exp}: {stats_dict}")