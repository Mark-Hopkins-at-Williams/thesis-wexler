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
import sys

CORPORA = {
    "train": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil/organized/train.eng",
    "dev": "/mnt/storage/swexler/thesis-wexler/examples/french-model_10_28_25/fr-en/dev.eng",
    "test": "/mnt/storage/swexler/thesis-wexler/examples/french-model_10_28_25/fr-en/test.eng",
}


class FileBasedLMData(IterableDataset):
    def __init__(self, filepath, max_length):
        self.filepath = filepath
        self.max_length = max_length

    def parse_line(self, line):
        return list(line.strip().encode("utf-8"))

    def __iter__(self):
        for line in open(self.filepath, "r"):
            tokens = self.parse_line(line)
            # print(f"TOKENS={chr(tokens[0])}")
            if len(tokens) < 2:
                input_ids = torch.tensor([tokens[0]], dtype=torch.long)
                target_ids = torch.tensor([], dtype=torch.long)
                print(f"input_ids={input_ids}")
                print(f"target_ids={target_ids}")
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
def evaluate(model, dataloader):
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
def compress(model, input_ids, target_ids, output_style=""):
    model.eval()
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)  # (B, T, vocab_size)
    preds = logits.argmax(dim=-1)  # (B, T)
    mask = target_ids == -100  # ignore padding positions

    correct = (
        preds == target_ids
    ) | mask  # our end goal here is to identify all the characters we mispredicted -- padding is kind of neutral so we leave it in {we care about the 0s here}
    text = []
    orig = []

    for i in range(len(correct)):  # each example in batch
        sentinel_active = True
        # If it's a dummy/empty line
        if torch.all(input_ids[i] == 0):
            print("ALL PADDING")
            text.append("")
            orig.append("")
            continue

        orig_line = [
            chr(input_ids[i][0].item())
        ]  # original sentence (here just 1st char)
        print(f"orig_line={orig_line}")

        compressed_line = [
            input_ids[i][0].item()
        ]  # compressed representation (here just 1st char)

        for j in range(len(correct[i])):  # each character in the example
            if (
                target_ids[i][j] >= 0
            ):  # filter out padding from condensed rep. (they are now lumped into correct)
                orig_line.append(
                    chr(target_ids[i][j].item())
                )  # add char to the original example
                # chr(100) gives the character w unicode code point 100
                if not correct[i][j]:
                    compressed_line.append(target_ids[i][j].item())
                    sentinel_active = True
                elif sentinel_active:  # we got it and need to add an emoji
                    compressed_line.append(128512)
                    # compressed_line[-1] += 256
                    if output_style == "-short":
                        sentinel_active = False  # replaces a whole string of correct predictions with just one emoji, rather than one emoji per character
        condensed_line_str = ""
        for item in compressed_line:
            if item != 128512:
                condensed_line_str += f"\\x{item:02x}"
            else:
                condensed_line_str = condensed_line_str + chr(128512)
        text.append(condensed_line_str)
        orig.append("".join(orig_line))
    # print(text)
    # print(orig)
    return text


# "wrapper method" of sorts for creating condensed representations
# initalizes model as best model from training then calls compress() repeatedly on batches of examples
def tokenize(model, loader, model_dir, output_dir, examples_type, output_style=""):
    checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("BEGINNING COMPRESSION")

    total_lines = get_total_lines(loader.dataset.filepath)
    total_batches = math.ceil(total_lines / loader.batch_size)

    with open(
        output_dir + "/compressed-" + examples_type + output_style + ".eng", "w"
    ) as writer:
        for input_ids, target_ids in tqdm(loader, total=total_batches):
            print("my input ids:")
            print(input_ids)
            print("my target ids:")
            print(target_ids)
            compressed = compress(model, input_ids, target_ids, output_style)
            for line in compressed:
                writer.write(f"{line}\n")


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(
    vocab_size,
    d_model=1024,
    nhead=16,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.1,
    max_len=1024,
).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
# print("BEGINNING TRAINING")
# train(
#     model,
#     loader,
#     optimizer,
#     val_loader,
#     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
#     training_steps=5000,
#     val_interval=500,
# )
print("BEGINNING TOKENIZATION")
# tokenize(
#     model,
#     val_loader,
#     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
#     output_dir="/mnt/storage/swexler/thesis-wexler/examples/one-char-examining",
#     examples_type="dev",
#     output_style="-short"
# )
# tokenize(
#     model,
#     test_loader,
#     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
#     output_dir="/mnt/storage/swexler/thesis-wexler/examples/english-byte-data",
#     examples_type="test"
# )

# # Long version: 1 emoji per correct guess
# tokenize(
#     model,
#     train_loader,
#     # output_style="-long",
#     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
#     output_dir="/mnt/storage/swexler/thesis-wexler/examples/english-byte-data",
#     examples_type="train"
# )

# Short version: 1 emoji per run of correct guesses
# tokenize(
#     model,
#     train_loader,
#     model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
#     output_dir="/mnt/storage/swexler/thesis-wexler/examples/english-data-compressed_1_11_26",
#     examples_type="train",
#     output_style="-short"
# )


dummy_dataset = FileBasedLMData(
    "examples/one-char-examining/one-char.eng",
    max_length=1024,
)
dummy_loader = DataLoader(
    dummy_dataset,
    batch_size=2,
    num_workers=0,
    collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
)
tokenize(
    model,
    dummy_loader,
    model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2",
    output_dir="examples/one-char-examining",
    examples_type="dummy",
    output_style="-short",
)

##### EVALUATION
# model_dir="/mnt/storage/swexler/thesis-wexler/models/autocomplete-v2"
# checkpoint = torch.load(
#     os.path.join(model_dir, "best_model.pt"), map_location="cpu"
# )
# model.load_state_dict(checkpoint["model_state_dict"])
# print(evaluate(model, val_loader))
