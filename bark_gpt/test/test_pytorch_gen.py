import torch
import torch.nn.functional as F
from bark_gpt.train.prepare_dataset import prepare_dataset
from bark_gpt.model.model import BarkGPT

dataset, dataset_ids, vocab_size, stoi, itos = prepare_dataset()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkGPT(vocab_size).to(device)


def generate(model, prompt_tokens, max_new_tokens=10, temperature=1.0):
    model.eval()
    tokens = prompt_tokens[:]
    for _ in range(max_new_tokens):
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x)
        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_id)
        if next_id == stoi["<EOS>"]:
            break
    return tokens


# example prompts
prompts = [
    ["User:", "speak", "Assistant:"],
]

out_ids = generate(
    model, [stoi[t] for t in prompts[0]], max_new_tokens=5, temperature=0.7
)
out_text = " ".join([itos[i] for i in out_ids])
print("=== Test Generation === ")
print(f"Prompt: {' '.join(prompts[0])} → {out_text}")
