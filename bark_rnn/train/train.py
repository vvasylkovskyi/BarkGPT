import os
import torch
import torch.nn as nn
from bark_rnn.train.prepare_dataset import prepare_dataset
from bark_rnn.model.model import BarkRNN

# Prepare dataset
dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_size = 32  # less memory
model = BarkRNN(vocab_size, hidden_size=hidden_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

batch_size = 16
seq_len = max(len(seq) for seq in dataset_ids)


# pad sequences
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
        inputs = batch[:, :-1]  # [B, T-1]
        targets = batch[:, 1:]  # [B, T-1]

        # RNN expects [B, T, input_size] = [B, T] with Embedding inside
        logits, _ = model(inputs)  # [B, T-1, vocab_size]

        # Reshape to [B*T, vocab_size] for cross-entropy
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(padded):.4f}")

# Save checkpoint
torch.save(
    {
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    },
    os.environ.get("MODEL_PATH_RNN", "bark_rnn_model.pt"),
)
