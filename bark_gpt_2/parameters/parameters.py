from dataclasses import dataclass
import torch
from transformers import AutoTokenizer


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


tokenizer = AutoTokenizer.from_pretrained("bark_gpt_2_tokenizer")
block_size = 32
n_layer = 2
n_head = 2
n_embd = 32
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
epochs = 3

training_parameters = TrainingParameters(
    batch_size=batch_size,
    accum_steps=accum_steps,
    effective_batch=effective_batch,
    lr_small=lr_small,
    lr_scaled=lr_scaled,
    epochs=epochs,
)

print(f"Training parameters: {training_parameters}")

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
