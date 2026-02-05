import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int  # was max_seq_len
    n_layer: int
    n_head: int
    n_embd: int

    # conventions
    dropout: float = 0.0
    bias: bool = True


class BarkGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.n_embd,
                    nhead=config.n_head,
                    dim_feedforward=4 * config.n_embd,  # GPT convention
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_layer)
            ]
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
