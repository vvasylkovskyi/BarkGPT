import random
import torch

vocab = [
    "<PAD>",
    "<EOS>",
    "woof",
    "arf",
    "ruff",
    "grrr",
    "User:",
    "Assistant:",
    "speak",
]


stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for i, tok in enumerate(vocab)}
vocab_size = len(vocab)


# Generate synthetic sequences
def make_sequence():
    num_barks = random.randint(2, 10)  # multiple barks
    barks = [random.choice(["woof", "arf", "ruff", "grrr"]) for _ in range(num_barks)]
    return ["User:", "speak", "Assistant:"] + barks + ["<EOS>"]


def prepare_dataset():
    dataset = [make_sequence() for _ in range(2000)]
    dataset_ids = [torch.tensor([stoi[t] for t in seq]) for seq in dataset]
    print(">>> Vocabulary Size:", vocab_size)
    print(">>> Sample sequence:", dataset[:2])
    print(">>> Sample sequence IDs:", dataset_ids[:2])

    return dataset, dataset_ids, vocab_size, stoi, itos


def main():
    return prepare_dataset()
