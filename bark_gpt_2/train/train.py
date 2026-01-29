import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from local_datasets.load_dataset import dataset
from transformers import AutoTokenizer
import time
from bark_gpt_2.ui.progress_bar import progress_bar
from bark_gpt_2.model.model import BarkGPT
from bark_gpt_2.parameters.parameters import training_parameters, model_config, device

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # important
vocab_size = tokenizer.vocab_size

### Training Parameters ###
batch_size = training_parameters.batch_size
# Gradient Accumulation
accum_steps = training_parameters.accum_steps  # 64 x 32 = 2048 effective batch
effective_batch = training_parameters.effective_batch


lr_small = training_parameters.lr_small

lr_scaled = training_parameters.lr_scaled
epochs = training_parameters.epochs

block_size = model_config.block_size


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        return_attention_mask=False,
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


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


lm_dataset = tokenized_dataset.map(group_texts, batched=True)

train_loader = DataLoader(
    lm_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

model = BarkGPT(model_config).to(device)
loss_fn = nn.CrossEntropyLoss()  # next-token prediction
optimizer = torch.optim.AdamW(model.parameters(), lr=training_parameters.lr_small)

for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()
    for step, batch in enumerate(train_loader):
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

        # Forward
        logits = model(inputs)

        # Compute loss
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

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
