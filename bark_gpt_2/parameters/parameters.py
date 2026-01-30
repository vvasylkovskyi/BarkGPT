from dataclasses import dataclass
import math
import torch
from transformers import AutoTokenizer
from local_datasets.load_dataset import dataset


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int  # was max_seq_len
    n_layer: int
    n_head: int
    n_embd: int


@dataclass
class TrainingParameters:
    batch_size: int = 64
    accum_steps: int = 32  # 64 x 32 = 2048 effective batch
    effective_batch: int = batch_size * accum_steps
    lr_small: float = 3e-4
    lr_scaled: float = lr_small * (effective_batch / batch_size)
    epochs: int = 3


@dataclass
class GenerationParameters:
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 10


tokenizer = AutoTokenizer.from_pretrained("bark_gpt_2_tokenizer")
block_size = 128
n_layer = 6
n_head = 2
n_embd = 256
vocab_size = tokenizer.vocab_size

model_config = GPTConfig(
    vocab_size=vocab_size,  # real BPE vocab
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
)

print(
    f"model parameters: vocab_size={vocab_size}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}"
)

batch_size = 16
accum_steps = 1
effective_batch = batch_size * accum_steps  # 64 x 32 = 2048 effective batch
lr_small = 3e-4  # Original learning rate for batch_size=16
lr_scaled = lr_small * (
    effective_batch / batch_size
)  # Scale LR linearly with effective batch
epochs = 2

training_parameters = TrainingParameters(
    batch_size=batch_size,
    accum_steps=accum_steps,
    effective_batch=effective_batch,
    lr_small=lr_small,
    lr_scaled=lr_scaled,
    epochs=epochs,
)

print(f"Training parameters: {training_parameters}")

generation_parameters = GenerationParameters(max_length=100, temperature=0.8, top_k=40)
print(f"Generation parameters: {generation_parameters}")

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        return_attention_mask=False,
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -----------------------------
# 3. Model parameters
# -----------------------------

# Transformer params (attention + MLP)
transformer_params = 12 * n_embd**2 * n_layer

# Token embeddings
token_emb_params = vocab_size * n_embd

# Positional embeddings
pos_emb_params = block_size * n_embd

# Total
model_params = transformer_params + token_emb_params + pos_emb_params

print(f"Estimated model parameters: {model_params:,} (~{model_params/1e6:.2f}M)")


# -----------------------------
# 4. Count total tokens
# -----------------------------
total_tokens = sum(len(x) for x in tokenized_dataset["input_ids"])
avg_tokens_per_example = total_tokens / len(tokenized_dataset)

print(f"Number of examples: {len(tokenized_dataset)}")
print(f"Total tokens: {total_tokens}")
print(f"Average tokens per example: {avg_tokens_per_example:.2f}")

# -----------------------------
# 5. Compute number of blocks
# -----------------------------
block_size = model_config.block_size
num_blocks = total_tokens // block_size
avg_tokens_per_block = block_size  # by design each block has block_size tokens

print(f"Number of blocks (block_size={block_size}): {num_blocks}")
print(f"Average tokens per block: {avg_tokens_per_block}")

# -----------------------------
# 6. Compute steps per epoch
# -----------------------------
batch_size = training_parameters.batch_size
accum_steps = training_parameters.accum_steps  # gradient accumulation

steps_per_epoch = math.ceil(num_blocks / batch_size / accum_steps)
tokens_per_step = batch_size * block_size  # tokens seen per step

print(f"Steps per epoch (with accumulation={accum_steps}): {steps_per_epoch}")
print(f"Tokens per optimizer step: {tokens_per_step}")

# -----------------------------
# 8. Chinchilla scaling check
# -----------------------------
tokens_needed = 20 * model_params
min_epochs = math.ceil(tokens_needed / total_tokens)
total_optimizer_steps = min_epochs * steps_per_epoch

print(f"Chinchilla tokens required: {tokens_needed:,}")
print(f"Minimum epochs to satisfy Chinchilla: {min_epochs}")
print(f"Total optimizer steps to reach Chinchilla: {total_optimizer_steps}")


# -----------------------------
# 9. Safety warning
# -----------------------------
if total_tokens < tokens_needed:
    print(
        "⚠ WARNING: Your dataset is smaller than Chinchilla recommends for this model size."
    )
    print("Consider training for more epochs or reducing model size.")
else:
    print("✅ Dataset size sufficient to satisfy Chinchilla rule in 1 epoch.")

if epochs < min_epochs:
    print(
        f"Warning: Current epochs ({epochs}) is less than Chinchilla minimum ({min_epochs}). Consider increasing epochs."
    )
else:
    print("Chinchilla training requirement satisfied.")
# -----------------------------
