import torch
import torch.nn as nn


class BarkGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=32, n_layer=2, n_head=2, seq_len=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head, dim_feedforward=128
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
