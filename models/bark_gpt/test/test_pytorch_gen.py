import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from bark_gpt.train.prepare_dataset import prepare_dataset
from bark_gpt.model.model import BarkGPT

dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkGPT(vocab_size, max_seq_len=max(len(seq) for seq in dataset_ids)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
batch_size = 16

# pad sequences to max length
seq_len = max(len(seq) for seq in dataset_ids)


def generate_gpt(
    model, stoi, itos, prompt: str, max_new_tokens=50, temperature=1.0, device="cpu"
):
    model.eval()
    tokens = prompt.split()
    input_ids = torch.tensor(
        [[stoi.get(t, stoi["<PAD>"]) for t in tokens]], device=device
    )

    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
        input_ids = next_id  # feed only last token

        if itos[next_id.item()] == "<EOS>":
            break

    output_tokens = [itos[i] for i in generated_ids[0].tolist()]
    return " ".join(output_tokens[len(tokens) :])


# ------------------------------
# Test generation
# ------------------------------
checkpoint = torch.load(
    os.environ.get("MODEL_PATH_GPT", "bark_gpt_model.pt"), map_location="cpu"
)
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

bark_gpt_model = BarkGPT(vocab_size, max_seq_len=seq_len)
bark_gpt_model.load_state_dict(checkpoint["model_state"])

prompt = "User: speak Assistant:"
output_text = generate_gpt(
    bark_gpt_model, stoi, itos, prompt, max_new_tokens=20, temperature=0.7
)
print("Generated:", output_text)
