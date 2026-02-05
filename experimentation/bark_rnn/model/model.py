import torch
import torch.nn as nn


class BarkRNN(nn.Module):
    def __init__(self, vocab_size, n_embd=32, hidden_size=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # The RNN core
        self.rnn = nn.RNN(input_size=n_embd, hidden_size=hidden_size, batch_first=True)

        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [B, T]
        hidden: [1, B, H]
        """
        x = self.token_emb(x)  # [B, T, n_embd]
        out, hidden = self.rnn(x, hidden)
        out = self.ln(out)  # [B, T, H]
        logits = self.head(out)  # [B, T, vocab]
        return logits, hidden
