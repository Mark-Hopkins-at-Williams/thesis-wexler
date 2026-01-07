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
    "train": "/mnt/storage/swexler/thesis-wexler/examples/french-data-18-mil/organized/train.eng",
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
            if len(tokens) < 2:
                continue  # can't make input-target pairs
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
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, target_ids


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


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # (B, T, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="sum"
        )

        # Compute predictions and accuracy
        preds = logits.argmax(dim=-1)  # (B, T)
        mask = target_ids != -100  # ignore padding positions
        correct = (preds == target_ids) & mask

        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()

    model.train()
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


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
    data_iter = iter(dataloader)
    for global_step in tqdm(range(training_steps)):
        total_loss = 0
        try:
            input_ids, target_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, target_ids = next(data_iter)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
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
def compress(model, input_ids, target_ids):
    model.eval()
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)  # (B, T, vocab_size)
    preds = logits.argmax(dim=-1)  # (B, T)
    mask = target_ids == -100  # ignore padding positions
    correct = (preds == target_ids) | mask
    text = []
    orig = []
    sentinel_active = True
    for i in range(len(correct)):
        orig_line = [chr(input_ids[i][0].item())]
        next_line = [input_ids[i][0].item()]
        for j in range(len(correct[i])):
            if target_ids[i][j] >= 0:
                orig_line.append(chr(target_ids[i][j].item()))
                if not correct[i][j]:
                    next_line.append(target_ids[i][j].item())
                    sentinel_active = True
                elif sentinel_active:
                    next_line[-1] += 256
                    sentinel_active = False
        text.append("".join([chr(code) for code in next_line]))
        orig.append("".join(orig_line))
    return text


def tokenize(model, loader, model_dir):
    checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open("compressed.en", "w") as writer:
        for input_ids, target_ids in tqdm(loader):
            # print(input_ids)
            # print(target_ids)
            compressed = compress(model, input_ids, target_ids)
            for line in compressed:
                writer.write(f"{line}\n")


# dataset = StreamingDatasetLMData('allenai/c4', 'en', 'train', max_length=1024)
dataset = FileBasedLMData(CORPORA["train"], max_length=1024)
loader = DataLoader(
    dataset,
    batch_size=64,
    collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
)
val_dataset = FileBasedLMData(CORPORA["dev"], max_length=1024)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
)
vocab_size = 256
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
train(model, loader, optimizer, val_loader, model_dir="experiments/autocomplete-v1")
# tokenize(model, loader, model_dir="autocomplete-v5")
