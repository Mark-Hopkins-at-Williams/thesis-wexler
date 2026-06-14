import argparse
import os
from pathlib import Path
from random import shuffle
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
import os

# so I can access tokenization.py in the folder above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenization import *


def reorganize(batch_size, root_dir, split, output_dir, user_tokenizer, max_len, eng_path, fr_path, shuffle_batches):
    """
    Reorganizes text files by sorting lines by token length and shuffling in batches.

    This function tokenizes each line of the English split file using a pretrained tokenizer,
    sorts all lines by their tokenized length, chunks them into batches, shuffles the batch
    order, and then reorders all split-related files accordingly. The reorganized files are
    saved to `output_dir`.

    IMPORTANT NOTICE: my changes that allow the code to work with files in different directories appears to have broken the functionality for when the files are in the same directory (ie only root_dir is specified). Until this is fixed, use eng_path and fr_path to manually specify the 2 files that must remain parallel

    Parameters
    ----------
    batch_size : int
        Number of lines per shuffled chunk. Shuffling is done at the chunk level.
    root_dir : Path
        Path to the directory containing input files named like `<split>.*`.
    split : str
        Prefix of the files to process (e.g., "train" for "train.en", "train.fr", etc.).
    output_dir : Path
        Path to the directory where reorganized files will be written. Must not exist prior to call.
    user_tokenizer : str
        The type of tokenizer used to determine length of examples.
        "sentinel" for SentinelTokenizer, "char" for CharacterTokenizer, "byte" for ByteTokenizer, "default"/nothing for BPE tokenizer.
    max_len : str
        The maximum line of a length that will be kept in the organized dataset.
    eng_path : str
        The location of the file with the english training set (path + file name) (if not in the same folder as the french training set)
    fr_path : str
        The location of the file with the french training set (path + file name) (if not in the same folder as the english training set)
    shuffle_batches : str
        "true" if batch order should be randomized
        "false" if batches should be ordered by length (ascending)

    Raises
    ------
    FileExistsError
        If `output_dir` already exists.
    """

    # os.mkdir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # files = list(root_dir.glob(f"*{split}.*"))
    # eng_file = f"compressed-{split}-short.eng"
    if eng_path != None:
      eng_file = eng_path
    else:
      eng_file = f"{split}.eng"
      # eng_file = root_dir / eng_file
    if fr_path != None:
      fr_file = fr_path
    else:
      fr_file = f"{split}.fr"
      # fr_file = root_dir / fr_file

    files = [eng_file, fr_file]
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = []

    # check files have same len
    with open(eng_file) as f:
      eng_len = sum(1 for _ in f)
    with open(fr_file) as f:
        fr_len = sum(1 for _ in f)
    assert eng_len == fr_len, f"File length mismatch: {eng_len} eng lines vs {fr_len} fr lines"

    # in case length should be calculated in a more specific manner
    if user_tokenizer == "sentinel":
        tokenizer = SentinelTokenizer()
    if user_tokenizer == "char":
        tokenizer = CharacterTokenizer()
    if user_tokenizer == "byte":
        tokenizer = lambda line: {"input_ids": [byte for byte in line.encode()]}
    if user_tokenizer == "compressed": # for getting length of compressed text
        tokenizer = lambda line: {"input_ids": [1 for _ in line]} 
        # make dummy array that has 1 for each char in the already compressed rep.


    with open(eng_file, encoding="utf-8") as reader:
        print("..Reading Eng..")
        for i, line in tqdm(enumerate(reader)):
            line = line.strip()
            tokens = tokenizer(line)["input_ids"]

            # Only keep sentences that won't be truncated
            if len(tokens) < max_len:
                lengths.append((len(tokens), i))
    line_nums_by_length = [line_num for _, line_num in sorted(lengths)]
    chunk_starts = [
        batch_size * k for k in range((len(line_nums_by_length) // batch_size) - 1)
    ]
    print("..Shuffling..")
    if shuffle_batches == "true":
      shuffle(chunk_starts)
    line_nums = []
    for start in chunk_starts:
        line_nums.extend(line_nums_by_length[start : start + batch_size])

    for filename in files:
        lang_code = filename.rsplit(".", 1)[-1]
        if fr_path == None: # default behavior: eng / fr in the same folder (ie no french path specified)
          output_path = output_dir / filename
          file_path = root_dir / filename
          # if lang_code == "eng":
          #   file_path = root_dir / f"{split}.{lang_code}"
          # if lang_code == "fr":
          #   file_path = root_dir / f"{split}.{lang_code}"
        else: # eng and fr paths specified
          output_path = output_dir / f"{split}.{lang_code}"
          if lang_code == "eng":
            file_path = eng_path
          if lang_code == "fr":
            file_path = fr_path
        lines = []

        print("..Final Reading/Writing..")
        with open(file_path, encoding="utf-8") as reader:
          for line in tqdm(reader):
              lines.append(line.strip())
        with open(output_path, "w", encoding="utf-8") as writer:
            for num in tqdm(line_nums):
                writer.write(lines[num] + "\n")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorders the sentences of a parallel corpus so that batched sentences have similar lengths."
    )
    parser.add_argument(
        "--in_dir", type=str, required=True, help="Directory with the original files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for storing the new, reordered files.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Desired batch size."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="default",
        help="Which tokenizer should be used when calculating the length of examples.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Maximum length an example can be to be included in organized file.",
    )
    parser.add_argument(
        "--eng_path",
        type=str,
        default=None,
        help="Path to the file with english text.",
    )
    parser.add_argument(
        "--fr_path",
        type=str,
        default=None,
        help="Path to the file with french text.",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=str,
        default="true",
        help="Should batches be shuffled by length?",
    )
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    reorganize(args.batch_size, in_dir, "train", out_dir, args.tokenizer, args.max_len, args.eng_path, args.fr_path, args.shuffle_batches)
