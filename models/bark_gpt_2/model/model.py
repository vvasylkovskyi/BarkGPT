import torch
import torch.nn as nn
from models.bark_gpt_2.parameters.parameters import GPTConfig


class GPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, ff_mult=4, dropout=0.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Separate Q, K, V projections for LoRA compatibility
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.o_proj = nn.Linear(n_embd, n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, ff_mult * n_embd),
            nn.GELU(),
            nn.Linear(ff_mult * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # Self-attention with causal mask
        x_res = x
        x = self.ln1(x)

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.o_proj(out)
        x = x_res + out

        # Feed-forward
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
        self.pos_emb = nn.Embedding(config.n_ctx, config.n_embd)

        self.layers = nn.ModuleList(
            [GPTBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        assert T <= self.config.n_ctx, "Sequence length exceeds model context size"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
