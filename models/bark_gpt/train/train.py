import os
import torch
import torch.nn as nn
from bark_gpt.train.prepare_dataset import prepare_dataset
from bark_gpt.model.model import BarkGPT, GPTConfig

dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"


config = GPTConfig(
    vocab_size=vocab_size,  # real BPE vocab
    block_size=32,
    n_layer=2,
    n_head=8,
    n_embd=8,
)

model = BarkGPT(config)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
batch_size = 16

# pad sequences to max length
seq_len = max(len(seq) for seq in dataset_ids)


def pad(seq):
    pad_len = seq_len - len(seq)
    if pad_len > 0:
        return torch.cat([seq, torch.tensor([stoi["<PAD>"]] * pad_len)])
    return seq


padded = torch.stack([pad(seq) for seq in dataset_ids]).to(device)

for epoch in range(5):
    perm = torch.randperm(len(padded))
    total_loss = 0
    for i in range(0, len(padded), batch_size):
        batch = padded[perm[i : i + batch_size]]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(padded):.4f}")

# Save model
torch.save(
    {
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    },
    os.environ.get("MODEL_PATH_GPT", "bark_gpt_model.pt"),
)
