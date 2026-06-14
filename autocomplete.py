import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import math
import torch.nn.functional as F
from torch.optim import Adam
import os
import json
import sys
from collections import defaultdict

CORPORA = {
    "train": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/train.eng",
    "dev": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/dev.eng",
    "test": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng",
}


# for autocomplete approach with 2 kinds of letters
STYLIZED_LETTERS = {
    "a": "ą",
    "A": "Ą",
    "b": "ḃ",
    "B": "Ḃ",
    "c": "č",
    "C": "Č",
    "d": "ď",
    "D": "Ď",
    "e": "ė",
    "E": "Ė",
    "f": "ƒ",
    "F": "Ƒ",
    "g": "ğ",
    "G": "Ğ",
    "h": "ȟ",
    "H": "Ȟ",
    "i": "į",
    "I": "Į",
    "j": "ǰ",
    "J": "ǰ", 
    "k": "ķ",
    "K": "Ķ",
    "l": "ľ",
    "L": "Ľ",
    "m": "ṁ",
    "M": "Ṁ",
    "n": "ň",
    "N": "Ň",
    "o": "ő",
    "O": "Ő",
    "p": "ṗ",
    "P": "Ṗ",
    "q": "ɋ",
    "Q": "Ɋ",
    "r": "ř",
    "R": "Ř",
    "s": "ș",
    "S": "Ș",
    "t": "ť",
    "T": "Ť",
    "u": "ů",
    "U": "Ů",
    "v": "ṽ",
    "V": "Ṽ",
    "w": "ẇ",
    "W": "Ẇ",
    "x": "ẋ",
    "X": "Ẋ",
    "y": "ẏ",
    "Y": "Ẏ",
    "z": "ž",
    "Z": "Ž",
    "0": "⓪",
    "1": "①",
    "2": "②",
    "3": "③",
    "4": "④",
    "5": "⑤",
    "6": "⑥",
    "7": "⑦",
    "8": "⑧",
    "9": "⑨",
    ".": "․",  # one dot leader
    ",": "‚",  # low comma
    "!": "ǃ",  # retroflex click
    "?": "¿",  # inverted question mark
    ":": "꞉",  # modifier letter colon
    ";": ";",  # Greek question mark
    "'": "ʹ",  # prime
    " ": "∙",
}

device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if __name__ == "__main__"
    else torch.device("cpu")
)


class FileBasedLMData(IterableDataset):
    def __init__(self, filepath, max_length):
        self.filepath = filepath
        self.max_length = max_length

    def parse_line(self, line):
        return list(line.strip().encode("utf-8"))

    def __iter__(self):
        for line in open(self.filepath, "r"):
            tokens = self.parse_line(line)
            if len(tokens) < 2:
                input_ids = torch.tensor([tokens[0]], dtype=torch.long)
                target_ids = torch.tensor([], dtype=torch.long)
                yield input_ids, target_ids  # nothing to predict but need to keep files parallel
                continue
            chunk = tokens[: self.max_length]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, target_ids


class StreamingDatasetLMData(IterableDataset):
    def __init__(self, name, lang, split, max_length):
        self.max_length = max_length
        self.ds = load_dataset(name, lang, split=split, streaming=True)

    def parse_line(self, line):
        return list(line.strip().encode("utf-8"))

    def __iter__(self):
        for document in self.ds:
            lines = document["text"].split("\n")
            for line in lines:
                tokens = self.parse_line(line)
                if len(tokens) < 2:
                    continue  # can't make input-target pairs
                chunk = tokens[: self.max_length]

                # all 'starting letters' (each character of an example minus the last) (each elem is hex byte)
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)

                # all 'predicted letters' (each character of an example beginning from the second) (each elem is hex byte)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, target_ids


# combining function - mainly does padding
def collate_causal_lm(batch, pad_token_id=0):
    inputs, targets = zip(*batch)
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    target_ids = pad_sequence(
        targets, batch_first=True, padding_value=-100
    )  # -100 is ignored in loss by default
    return input_ids, target_ids


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_len
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.positional_encoding(x)

        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float("-inf"))

        x = self.transformer(x, mask=causal_mask)  # Only self-attention
        return self.lm_head(x)  # (B, T, vocab_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def get_total_lines(filepath):
    with open(filepath, "rb") as f:
        return sum(1 for _ in f)


@torch.no_grad()
def evaluate(model, dataloader, device=device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(
            input_ids
        )  # (B, T, vocab_size) = batch size rows, # of tokens length, vocab size depth (logit for each letter)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="sum"
        )

        # Compute predictions and accuracy
        preds = logits.argmax(
            dim=-1
        )  # (B, T) = batch size rows, # of tokens length where each item is most probable char
        mask = (
            target_ids != -100
        )  # an example doesn't know its padded --> ignore padding positions
        correct = (
            preds == target_ids
        ) & mask  # Boolean 2d array of each char where item is True if prediction is correct
        # & mask part allows it to ignore masked tokens

        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()

    model.train()
    avg_loss = total_loss / total_tokens
    accuracy = (
        total_correct / total_tokens
    )  # % of next tokens the model predicted correctly
    return avg_loss, accuracy


# does training loop and updates saved model if applicable during evaluation periods
def train(
    model,
    dataloader,
    optimizer,
    val_dataloader=None,
    val_interval=1000,
    training_steps=500000,
    model_dir=None,
):
    model.train()
    best_val_loss = None
    data_iter = iter(dataloader)  # dataloader is batched
    total_loss = 0
    for global_step in tqdm(range(training_steps)):

        try:
            input_ids, target_ids = next(data_iter)  # these are 2d bc they are batched
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, target_ids = next(data_iter)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1)
        )  # loss between predictions and target IDs {-log prob of correct token} --> doing for a whole sentence at once
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if val_dataloader is not None and global_step % val_interval == 0:
            val_loss, val_acc = evaluate(model, val_dataloader)
            if model_dir is not None and (
                best_val_loss is None or val_loss < best_val_loss
            ):
                print("Saving new best model.")
                best_val_loss = val_loss
                os.makedirs(model_dir, exist_ok=True)
                checkpoint_path = os.path.join(model_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "step": global_step,
                    },
                    checkpoint_path,
                )

            print(
                f"[Step {global_step}] Validation loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}"
            )
            sys.stdout.flush()


@torch.no_grad()
def compress(
    model,
    input_ids,
    target_ids,
    prediction_threshold,
    device="cpu",
    output_style="",
    prediction_mode="margin",
    autocomplete_mode="default",
    stylized_letters=STYLIZED_LETTERS,
):
    model.eval()

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    logits = model(input_ids)  # (B, T, vocab_size)

    id_for_unknown = -1  # What to return if threshold isn't met for any token
    temperature = 1

    preds = get_predictions(
        logits,
        prediction_threshold,
        mode=prediction_mode,
        temperature=1,
        id_for_unknown=-1,
    )

    # preds = logits.argmax(dim=-1)  # (B, T)
    mask = target_ids == -100  # ignore padding positions

    correct = (  # row is example col is tok
        preds == target_ids
    ) | mask  # our end goal here is to identify all the characters we mispredicted -- padding is kind of neutral so we leave it in {we care about the 0s here}
    condensed_batch = []

    for i in range(len(correct)):  # each example in batch
        sentinel_active = True
        # If it's a dummy/empty line
        if torch.all(input_ids[i] == 0):
            print("ALL PADDING")
            condensed_batch.append("")
            continue

        condensed_line = [
            input_ids[i][0].item()
        ]  # compressed representation (here just 1st char)

        if autocomplete_mode == "store_num_chars_autocompleted":
            num_correct = 0
            is_last = False
            for j in range(len(correct[i])):  # each character in the example
                if (
                    target_ids[i][j] >= 0
                ):  # filter out padding from condensed rep. (padding is now lumped into correct)
                    if not correct[i][j]:
                        condensed_line.append(target_ids[i][j].item())
                        num_correct = 0
                    else:  # correct prediction
                        num_correct += 1
                        if (
                            j == len(correct[i]) - 1
                        ):  # we are at the end of the longest example in batch
                            is_last = True
                        else:
                            next_is_padding = target_ids[i][j + 1] < 0
                            next_is_incorrect = not correct[i][j + 1]
                            if next_is_padding or next_is_incorrect:
                                is_last = True

                        if is_last:  # we only add emojis at the end of a run
                            if num_correct >= 8:
                              condensed_line.append(128512 + 8)
                            else:
                              condensed_line.append(128512 + num_correct)
                            num_correct = 0
                            is_last = False

        if autocomplete_mode == "no_autocomplete_chars":
            for j in range(len(correct[i])):  # each character in the example
                if (
                    target_ids[i][j] >= 0
                ):  # filter out padding from condensed rep. (padding is now lumped into correct)
                    if not correct[i][j]:
                        condensed_line.append(target_ids[i][j].item())
        if autocomplete_mode == "two_types_english_chars": # every hint except last one in chunk is stylized
            for j in range(len(correct[i])):  # each character in the example
                if (
                    target_ids[i][j] >= 0
                ):  # filter out padding from condensed rep. (padding is now lumped into correct)
                    if not correct[i][j]:
                        if j > 0 and not correct[i][j - 1]: # if the previous letter was also a hint
                            if chr(condensed_line[-1]) in stylized_letters: 
                                replacement_char = stylized_letters[ # replace the previous hint with stylized version
                                    chr(condensed_line[-1])
                                ]
                                condensed_line[-1] = ord(replacement_char) # condensed_line stores ints
                        condensed_line.append(target_ids[i][j].item())
        if autocomplete_mode == "default":  # default approach
            for j in range(len(correct[i])):  # each character in the example
                if (
                    target_ids[i][j] >= 0
                ):  # filter out padding from condensed rep. (padding is now lumped into correct)
                    if not correct[i][j]:
                        condensed_line.append(target_ids[i][j].item())
                        sentinel_active = True
                    elif sentinel_active:  # we got it and need to add an emoji
                        condensed_line.append(128512)
                        if output_style == "-short":
                            sentinel_active = False  # replaces a whole string of correct predictions with just one emoji, rather than one emoji per character

        # CONSTRUCTING COMPRESSED REPRESENTATION IN THE CORRECT FORMAT
        condensed_line_str = ""
        for item in condensed_line:
            if item > 0 and item < 256:
                condensed_line_str += f"\\x{item:02x}"  # format hex string correctly
            else:
                condensed_line_str = condensed_line_str + chr(item)
        condensed_batch.append(condensed_line_str)
    # print(condensed_batch)
    return condensed_batch


def get_predictions(
    logits, prediction_threshold, mode, temperature=1, id_for_unknown=-1
):
    if mode == "top_pred":
        probs = F.softmax(logits / temperature, dim=-1)  # converting each logit to %s
        probs, preds = torch.max(
            probs, dim=-1
        )  # one matrix for the each char prediction and another matrix for "confidence" of each prediction
        preds = torch.where(
            probs >= prediction_threshold, preds, torch.tensor(id_for_unknown)
        )  # (B,T)
        # Compare all of probs to all of threshold. If the prediction is sufficiently confident, the prediction is used for the element. Otherwise, id_for_unknown used
        return preds
    else:
        probs = F.softmax(logits / temperature, dim=-1)  # converting each logit to %s
        top_probs, top_preds = torch.topk(
            probs, k=2, dim=-1
        )  # # one matrix for top 2 char predictions and another matrix for "confidence" of these predictions

        top_guess_confidence = top_probs[:, :, 0]  # all probs of top guess
        second_guess_confidence = top_probs[:, :, 1]  # all probs of second guess
        top_guess = top_preds[:, :, 0]  # the actual top guesses

        margin = (
            top_guess_confidence - second_guess_confidence
        )  # gap between 1st and 2nd guesses
        preds = torch.where(
            margin >= prediction_threshold, top_guess, torch.tensor(id_for_unknown)
        )
        # If the prediction is sufficiently confident, the prediction is used for the element. Otherwise, id_for_unknown used
        return preds


# "wrapper method" of sorts for creating condensed representations
# initalizes model as best model from training then calls compress() repeatedly on batches of examples
def tokenize(
    model,
    loader,
    model_dir,
    output_dir,
    examples_type,
    prediction_threshold=0.8,
    output_style="",
    prediction_mode="margin",
    autocomplete_mode="default",
):
    checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("--beginning compression")

    total_lines = get_total_lines(loader.dataset.filepath)
    total_batches = math.ceil(total_lines / loader.batch_size)

    if output_style != "":
        output_style = "-" + output_style

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"compressed-{examples_type}{output_style}.eng"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as writer:
        for input_ids, target_ids in tqdm(loader, total=total_batches):

            compressed = compress(
                model,
                input_ids,
                target_ids,
                prediction_threshold,
                device,
                output_style,
                prediction_mode,
                autocomplete_mode,
            )
            for line in compressed:
                writer.write(f"{line}\n")



if __name__ == "__main__":
    # dataset = StreamingDatasetLMData('allenai/c4', 'en', 'train', max_length=1024)

    train_dataset = FileBasedLMData(CORPORA["train"], max_length=1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    )
    val_dataset = FileBasedLMData(CORPORA["dev"], max_length=1024)
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    )
    test_dataset = FileBasedLMData(CORPORA["test"], max_length=1024)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    )
    vocab_size = 263  # 256 + bos + eos + pad + mask + eng + fra + autocomplete

    model = DecoderOnlyTransformer(
        vocab_size,
        d_model=1024,
        nhead=16,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_len=1024,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # print("BEGINNING TRAINING")
    # separate_dataset = FileBasedLMData("/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.2007.en.shuffled", max_length=1024)
    # separate_dataset_loader = DataLoader(
    #     separate_dataset,
    #     batch_size=64,
    #     num_workers=0,
    #     collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    # )
    # train(
    #     model,
    #     separate_dataset_loader,
    #     optimizer,
    #     val_loader,
    #     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v5",
    #     training_steps=50000,
    #     val_interval=500
    # )

    """ 
    autocomplete modes = store_num_chars_autocompleted
                         no_autocomplete_chars
                         two_types_english_chars
                         default

    prediction thresholds testing:
    margin   = 0.1, 0.3, 0.5
    top_pred = 0.3, 0.5, 0.7
    """

    specifications = {
        "source_data": os.path.dirname(CORPORA["train"]),
        "compression_model": "/mnt/storage/swexler/thesis-wexler/models/autocomplete-v5",
        "prediction_threshold": 0.3,  # float within (0,1)
        "prediction_mode": "top_pred",  # margin or top_pred (absolute)
        "output_style": "short",  # short or long
        "date_compressed": "4_21_26",
        "autocomplete_mode": "no_autocomplete_chars"
    }
    desired_output_dir = (
        "/mnt/storage/swexler/thesis-wexler/examples/english-data-compressed_"
        + specifications["date_compressed"]
        + "-"
        + str(specifications["prediction_threshold"])
        + "-"
        + specifications["prediction_mode"]
        + "-"
        + specifications["autocomplete_mode"]
    )

    ## Making json with how we compressed the files
    os.makedirs(desired_output_dir, exist_ok=True)
    documentation_path = os.path.join(desired_output_dir, "specs.json")
    with open(documentation_path, "w") as f:
        json.dump(specifications, f, indent=4)

    print("BEGINNING TOKENIZATION")
    tokenize(
        model,
        val_loader,
        model_dir=specifications["compression_model"],
        output_dir=desired_output_dir,
        examples_type="dev",
        prediction_threshold=specifications["prediction_threshold"],
        prediction_mode=specifications["prediction_mode"],
        output_style=specifications["output_style"],
        autocomplete_mode=specifications["autocomplete_mode"],
    )
    tokenize(
        model,
        test_loader,
        model_dir=specifications["compression_model"],
        output_dir=desired_output_dir,
        examples_type="test",
        prediction_threshold=specifications["prediction_threshold"],
        prediction_mode=specifications["prediction_mode"],
        output_style=specifications["output_style"],
        autocomplete_mode=specifications["autocomplete_mode"],
    )

    tokenize(
        model,
        train_loader,
        model_dir=specifications["compression_model"],
        output_dir=desired_output_dir,
        examples_type="train",
        prediction_threshold=specifications["prediction_threshold"],
        prediction_mode=specifications["prediction_mode"],
        output_style=specifications["output_style"],
        autocomplete_mode=specifications["autocomplete_mode"]
    )

    """
    BELOW HERE IS DUMMY DATASET 
    """
    # ONE CHAR DUMMY
    # dummy_dataset = FileBasedLMData(
    #     "examples/one-char-examining/one-char.eng",
    #     max_length=1024,
    # )
    # dummy_loader = DataLoader(
    #     dummy_dataset,
    #     batch_size=3,
    #     num_workers=0,
    #     collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    # )
    # tokenize(
    #     model,
    #     dummy_loader,
    #     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v5",
    #     output_dir="/mnt/storage/swexler/thesis-wexler/examples/one-char-examining",
    #     examples_type="dummy",
    #     prediction_threshold=0.3,
    #     output_style="short",
    #     prediction_mode="top_pred",
    #     autocomplete_mode="default"
    # )

    """ 
    modes = store_num_chars_autocompleted
            no_autocomplete_chars
            two_types_english_chars
            default
    """

    ##### EVALUATION
    # model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2"
    # checkpoint = torch.load(
    #     os.path.join(model_dir, "best_model.pt"), map_location="cpu"
    # )
    # model.load_state_dict(checkpoint["model_state_dict"])
    # print(evaluate(model, val_loader))
