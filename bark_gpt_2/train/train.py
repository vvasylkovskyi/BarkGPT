import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import time
from bark_gpt_2.ui.progress_bar import progress_bar
from bark_gpt_2.model.model import BarkGPT
from bark_gpt_2.parameters.parameters import (
    training_parameters,
    model_config,
    device,
    tokenized_dataset,
)
from datasets import load_from_disk
from logger.logger import Logger

logger = Logger("train")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # important

### Training Parameters ###
batch_size = training_parameters.batch_size
# Gradient Accumulation
accum_steps = training_parameters.accum_steps  # 64 x 32 = 2048 effective batch
effective_batch = training_parameters.effective_batch


lr_small = training_parameters.lr_small

lr_scaled = training_parameters.lr_scaled
epochs = training_parameters.epochs

block_size = model_config.block_size


CKPT_PATH = "checkpoints/barkgpt_ckpt.pt"
os.makedirs("checkpoints", exist_ok=True)


def save_checkpoint(
    epoch: int, step_in_epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer
):
    logger.info(
        f"Saving checkpoint at epoch {epoch}, step_in_epoch {step_in_epoch}, global_step {global_step}"
    )
    tmp = f"{CKPT_PATH}.tmp"
    torch.save(
        {
            "epoch": epoch,
            "step": step_in_epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "vocab_size": model_config.vocab_size,
            "max_seq_len": model_config.block_size,
        },
        tmp,
    )
    os.replace(tmp, CKPT_PATH)


def sanity_loss(model, loss_fn, sample_batch):
    model.eval()  # disable dropout / training effects
    with torch.no_grad():  # do NOT compute gradients
        inputs = sample_batch[:, :-1]
        targets = sample_batch[:, 1:]
        logits = model(inputs)
        loss = loss_fn(
            logits.reshape(-1, model_config.vocab_size), targets.reshape(-1)
        ).item()
    model.train()  # restore training mode
    return loss


def collate_fn(batch):
    # batch is a list of dicts: [{"input_ids": [...]}, ...]
    input_ids = [torch.tensor(item["input_ids"], device=device) for item in batch]
    # pad sequences to the max length in this batch
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return {"input_ids": input_ids}


def group_texts(examples):
    # Flatten all token IDs
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_len]

    # Split into blocks of `block_size`
    input_ids = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": input_ids}


def load_grouped_dataset_from_cache():
    CACHE_DIR = "cache/lm_dataset"

    if os.path.exists(CACHE_DIR):
        logger.success("Loading tokenized + grouped dataset from disk")
        lm_dataset = load_from_disk(CACHE_DIR)

    else:
        logger.info("Building tokenized + grouped dataset (one-time)")
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
        )

        lm_dataset.save_to_disk(CACHE_DIR)
        logger.info("Dataset saved to disk")

    return lm_dataset


lm_dataset = load_grouped_dataset_from_cache()

train_loader = DataLoader(
    lm_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

model = BarkGPT(model_config).to(device)
loss_fn = nn.CrossEntropyLoss()  # next-token prediction
optimizer = torch.optim.AdamW(model.parameters(), lr=training_parameters.lr_small)

start_epoch = 0
global_step = 0
start_step = 0

checkpoint_interval = training_parameters.checkpoint_interval

## Resume from checkpoint if available
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_epoch = ckpt["epoch"]
    start_step = ckpt["step"]
    global_step = ckpt.get("global_step", 0)
    logger.success(f"Resumed from epoch {start_epoch}, step {start_step}")


def train(global_step: int, start_epoch: int, start_step: int):
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            # skip until we reach the saved step
            if epoch == start_epoch and step < start_step:
                continue

            # Minimal progress output
            progress_bar(
                step,
                total=len(train_loader),
                start_time=start_time,
                prefix=f"Epoch {epoch+1}",
            )
            # Convert to tensor
            input_ids = batch["input_ids"]

            # Next-token prediction
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # -------------------------
            # Forward & backward
            # -------------------------
            logits = model(inputs)
            loss = loss_fn(
                logits.reshape(-1, model_config.vocab_size), targets.reshape(-1)
            )
            loss = loss / accum_steps  # scale for accumulation
            loss.backward()

            # Early NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN/Inf detected at epoch {epoch}, step {step}, global_step {global_step}"
                )
                return  # abort training

            # # Optional gradient clipping for stability
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_loss += loss.item() * accum_steps  # scale back for logging

            # -------------------------
            # Step optimizer every accum_steps
            # -------------------------
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % checkpoint_interval == 0:
                    ckpt_loss = sanity_loss(model, loss_fn, input_ids)  # small batch
                    logger.info(
                        f"Checkpoint at step {global_step}: sanity loss={ckpt_loss:.4f}"
                    )
                    save_checkpoint(epoch, step, model, optimizer)

        logger.success(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        save_checkpoint(epoch + 1, 0, model, optimizer)


train(global_step, start_epoch, start_step)
logger.success("Training complete.")
# save model weights
torch.save(
    {
        "model_state": model.state_dict(),
        "vocab_size": model_config.vocab_size,
        "max_seq_len": model_config.block_size,
    },
    "bark_gpt_2_model.pt",
)

tokenizer.save_pretrained("bark_gpt_2_tokenizer")
