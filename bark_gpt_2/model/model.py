import torch
import torch.nn as nn
from bark_gpt_2.parameters.parameters import GPTConfig


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, ff_mult * n_embd),
            nn.GELU(),
            nn.Linear(ff_mult * n_embd, n_embd),
        )

    def forward(self, x):
        B, T, C = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x_res = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + x_res

        x_res = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x + x_res
        return x


class BarkGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.layers = nn.ModuleList(
            [GPTBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
