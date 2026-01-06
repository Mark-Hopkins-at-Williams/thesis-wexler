import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e, f):
        Q, K, V = e, f, f
        scores = (Q @ K.transpose(-2, -1)) / (e.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        # print(weights)
        output = torch.matmul(weights, V)
        # print(output)
        return output, weights


if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 3, 5
    src = torch.randn(batch_size, seq_len, embed_dim)
    tgt = src.clone()
    tgt = tgt.flip(dims=[1])
    tgt = tgt * 2

    print(src)
    print(tgt)

    attn = SimpleAttention()
    out, weights = attn(src, tgt)

    token_scores = 1 - F.cosine_similarity(src, out, dim=-1)  # [seq_len]
    loss = token_scores.mean()
    print(loss)

    print("Output shape:", out.shape)  # (2, 5, 16)
    print(out)
    # print("Attention weights shape:", weights.shape)  # (2, 5, 5)

    print(token_scores)
