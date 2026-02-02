from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Dict, cast
import torch
from transformers import AutoTokenizer
from local_datasets.load_dataset_small import dataset
from logger.logger import Logger
from bark_gpt_2.ui.progress_bar import progress_bar
from datasets import DatasetDict
from datasets import load_from_disk

logger = Logger("parameters")


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
    checkpoint_interval: int = 1000  # Save checkpoint every N steps


@dataclass
class GenerationParameters:
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 10


def safe_get_dataset_split(ds: DatasetDict, split_name: str):
    """Safe access: check if dataset has splits or use it directly"""
    if hasattr(ds, "keys") and split_name in ds.keys():
        return ds[split_name]
    return ds


def get_total_tokens():
    """Count total tokens in the dataset"""
    META_PATH = "cache/lm_dataset_meta/meta.json"
    total_tokens = 0
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            total_tokens = json.load(f)["total_tokens"]
    else:
        for i, x in enumerate(input_ids):
            total_tokens += len(x)
            if i % 1000 == 0 or i + 1 == total:
                progress_bar(
                    i,
                    total=total,
                    start_time=start_time,
                    prefix="Counting tokens",
                )
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        with open(META_PATH, "w") as f:
            json.dump({"total_tokens": total_tokens}, f)
    return total_tokens


tokenizer = AutoTokenizer.from_pretrained("gpt2")
block_size = 128
n_layer = 4
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

logger.info(
    f"model parameters: vocab_size={vocab_size}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}"
)

batch_size = 16
accum_steps = 1
effective_batch = batch_size * accum_steps  # 64 x 32 = 2048 effective batch
lr_small = 1e-4  # Original learning rate for batch_size=16
lr_scaled = lr_small * (
    effective_batch / batch_size
)  # Scale LR linearly with effective batch
epochs = 1
checkpoint_interval = 500  # Save checkpoint every N steps

training_parameters = TrainingParameters(
    batch_size=batch_size,
    accum_steps=accum_steps,
    effective_batch=effective_batch,
    lr_small=lr_small,
    lr_scaled=lr_scaled,
    epochs=epochs,
    checkpoint_interval=checkpoint_interval,
)

logger.info(f"Training parameters: {training_parameters}")

generation_parameters = GenerationParameters(max_length=100, temperature=0.8, top_k=40)
logger.info(f"Generation parameters: {generation_parameters}")

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info(f"Using device: {device}")


def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tokenize a batch of examples.
    Args:
        batch: A dictionary with keys like 'text'.
    Returns:
        A dictionary with tokenized outputs, e.g., {'input_ids': [...]}
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        return_attention_mask=False,
    )


def load_tokenized_dataset_from_cache() -> DatasetDict:
    """
    Load the tokenized dataset from disk cache if available.
    Otherwise, tokenize and save to disk.

    Returns:
        tokenized_dataset: HuggingFace Dataset object
    """
    TOKENIZED_CACHE = "cache/tokenized_dataset"

    if os.path.exists(TOKENIZED_CACHE):
        tokenized_dataset: DatasetDict = cast(
            DatasetDict, load_from_disk(TOKENIZED_CACHE)
        )
        logger.success("Loaded tokenized dataset from disk cache")
    else:
        tokenized_dataset: DatasetDict = dataset.map(
            tokenize, batched=True, remove_columns=["text"]
        )
        tokenized_dataset.save_to_disk(TOKENIZED_CACHE)
        logger.info("Tokenized dataset saved to disk cache")

    return tokenized_dataset


tokenized_dataset: DatasetDict = load_tokenized_dataset_from_cache()
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

logger.info(
    f"Transformer parameters: {transformer_params:,} (~{transformer_params/1e6:.2f}M)"
)
logger.info(
    f"Token embedding parameters: {token_emb_params:,} (~{token_emb_params/1e6:.2f}M)"
)
logger.info(
    f"Positional embedding parameters: {pos_emb_params:,} (~{pos_emb_params/1e6:.2f}M)"
)
logger.info(f"Estimated model parameters: {model_params:,} (~{model_params/1e6:.2f}M)")


# -----------------------------
# 4. Count total tokens
# -----------------------------
start_time = time.time()
total_tokens = 0

input_ids = safe_get_dataset_split(tokenized_dataset, "train")["input_ids"]
total = len(input_ids)
total_tokens = get_total_tokens()

logger.info(f"Total tokens: {total_tokens:,}")
avg_tokens_per_example = total_tokens / len(
    safe_get_dataset_split(tokenized_dataset, "train")
)

logger.info(
    f"Number of examples: {len(safe_get_dataset_split(tokenized_dataset, 'train')):,}"
)
logger.info(f"Total tokens: {total_tokens:,}")
logger.info(f"Average tokens per example: {avg_tokens_per_example:.2f}")

# -----------------------------
# 5. Compute number of blocks
# -----------------------------
block_size = model_config.block_size
num_blocks = total_tokens // block_size
avg_tokens_per_block = block_size  # by design each block has block_size tokens

logger.info(f"Number of blocks (block_size={block_size}): {num_blocks:,}")
logger.info(f"Average tokens per block: {avg_tokens_per_block:,}")

# -----------------------------
# 6. Compute steps per epoch
# -----------------------------
batch_size = training_parameters.batch_size
accum_steps = training_parameters.accum_steps  # gradient accumulation

steps_per_epoch = math.ceil(num_blocks / batch_size / accum_steps)
tokens_per_step = batch_size * block_size  # tokens seen per step

logger.info(f"Steps per epoch (with accumulation={accum_steps}): {steps_per_epoch:,}")
logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")

# -----------------------------
# 8. Chinchilla scaling check
# -----------------------------
tokens_needed = 20 * model_params
min_epochs = math.ceil(tokens_needed / total_tokens)
total_optimizer_steps = min_epochs * steps_per_epoch

logger.info(f"Chinchilla tokens required: {tokens_needed:,}")
logger.info(f"Minimum epochs to satisfy Chinchilla: {min_epochs}")
logger.info(f"Total optimizer steps to reach Chinchilla: {total_optimizer_steps:,}")


# -----------------------------
# 9. Safety warning
# -----------------------------
if total_tokens < tokens_needed:
    logger.warning(
        "⚠ WARNING: Your dataset is smaller than Chinchilla recommends for this model size."
    )
    logger.warning("Consider training for more epochs or reducing model size.")
else:
    logger.info("✅ Dataset size sufficient to satisfy Chinchilla rule in 1 epoch.")

if epochs < min_epochs:
    logger.warning(
        f"Warning: Current epochs ({epochs}) is less than Chinchilla minimum ({min_epochs}). Consider increasing epochs."
    )
else:
    logger.info("Chinchilla training requirement satisfied.")
# -----------------------------
