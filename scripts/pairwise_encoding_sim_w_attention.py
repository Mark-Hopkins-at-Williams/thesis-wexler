import sys
import os
sys.path.append(os.path.abspath(".."))
import json
import torch
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import faiss
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tokenization import NllbTokenizer, HuggingfaceTokenizer
from attention import SimpleAttention

USE_CUDA = torch.cuda.is_available()
import torch.nn.functional as F

def logger(s):
    sys.stderr.write(f'{s}\n')
    sys.stderr.flush()


def compute_language_embeddings(model, tokenized_data, lang_codes):
    """
    Collect fine-grained token-level embeddings for the source language of each batch.
    Returns a dict: lang_code -> list of [seq_len, hidden_dim] numpy arrays
    """
    model.eval()
    encoder = model.model.encoder
    embeddings = defaultdict(list)

    with torch.no_grad():
        batch = tokenized_data.next_batch()
        while batch is not None:
            x, _, src_lang, _ = batch  # ignore target

            # Move to device
            x = x.to(model.device)

            # Get encoder outputs
            x_enc = encoder(**x).last_hidden_state[0]  # [seq_len, hidden_dim]

            # Store embeddings for source language only
            embeddings[lang_codes[src_lang]].append(x_enc.cpu().numpy())

            batch = tokenized_data.next_batch()

    return dict(embeddings)



def compute_attention_heatmap(embeddings):
    """
    embeddings: dict lang_code -> list of [seq_len, hidden_dim] arrays
    Returns a matrix of average cosine similarity (attention-based) between languages.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    lang_list = list(embeddings.keys())

    n = len(lang_list)
    heatmap = np.zeros((n, n))

    attn = SimpleAttention()

    for i, lang_i in tqdm(enumerate(lang_list)):
        for j, lang_j in enumerate(lang_list):
            scores = []
            # Compare sentence by sentence (assume same number of sentences)
            for k in range(min(len(embeddings[lang_i]), len(embeddings[lang_j]))):
                E = torch.tensor(embeddings[lang_i][k])  # [seq_len, hidden_dim]
                G = torch.tensor(embeddings[lang_j][k])
                out, weights = attn(E, G)  # [seq_len, hidden_dim]

                # Compute cosine similarity per token between E and attended context
                token_scores = F.cosine_similarity(E, out, dim=-1)  # [seq_len]
                scores.append(token_scores.mean().item())

            heatmap[i, j] = np.mean(scores)

    return heatmap, lang_list


def plot_clustermap(matrix, labels, out_file="attention_heatmap.png"):
    """
    Same as before, just updated title
    """
    import pandas as pd
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    g = sns.clustermap(
        df,
        cmap="viridis",
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
    )

    plt.suptitle("Attention Avg Cosine Similarity (Fine-Grained)", y=1.02)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved clustermap to {out_file}")



def main():
    # Load config JSON
    with open("config.json") as f:
        config = json.load(f)

    # Extract language codes
    lang_codes = {
        (c, k): config['corpora'][c][k]['lang_code']
        for c in config['corpora'] for k in config['corpora'][c]
    }
    LANGS = list(lang_codes.values())


    # Load model
    logger('loading model...')
    
    model_name = config["finetuning_parameters"]["base_model"]
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()

    # Load dev data and tokenize
    logger('tokenizing dev data...')
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    
    if model_name == "facebook/nllb-200-distilled-600M":  
        tokenizer = NllbTokenizer("600M", max_length=128) # set max length?
    elif model_name == "facebook/nllb-200-distilled-1.3B": 
        tokenizer = NllbTokenizer("1.3B", max_length=128)
    else:
        tokenizer = HuggingfaceTokenizer(model_name, max_length=128)
    
    
    tokenized_dev = TokenizedMixtureOfBitexts(dev_data, tokenizer,
                                              lang_codes=lang_codes, permutation_map={})
    
    # Compute fine-grained embeddings
    logger('computing embeddings...')
    embeddings = compute_language_embeddings(model, tokenized_dev, lang_codes)
    #print(embeddings)


    # Compute FAISS heatmap
    logger('computing heatmap...')
    heatmap, lang_list = compute_attention_heatmap(embeddings)

    # Plot heatmap
    plot_clustermap(heatmap, lang_list, out_file="attention_heatmap.png")
    logger("completed!")

if __name__ == "__main__":
    main()
