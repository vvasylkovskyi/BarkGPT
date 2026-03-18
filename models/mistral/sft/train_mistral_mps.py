# train_mistral_mps.py
# Install: pip install transformers datasets torch accelerate peft bitsandbytes

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# Set device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the tokenizer and model
model_name = "mistralai/Mistral-7B-v0.1"
print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading model from {model_name}...")
# Load model in float16 to save memory on MPS
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": device},  # Load directly to MPS
    low_cpu_mem_usage=True,
)

# Set padding token id in model config
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,  # Rank - higher = more parameters but better quality
    lora_alpha=32,  # Scaling factor
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # Uncomment below for more thorough fine-tuning (uses more memory)
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Load the processed datasets
print("Loading processed datasets...")
train_dataset = load_from_disk("./data/oasst_mistral_train")
val_dataset = load_from_disk("./data/oasst_mistral_val")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"First training sample input_ids length: {len(train_dataset[0]['input_ids'])}")
# Optional: Use a subset for faster testing
# train_dataset = train_dataset.select(range(1000))
# val_dataset = val_dataset.select(range(100))

# 4. Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # We're doing causal LM, not masked LM
)

# 5. Training arguments
output_dir = "./mistral-7b-instruct-oasst"
print(f"Output directory: {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjust based on your Mac's memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
    # Learning rate and schedule
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    # Optimization
    optim="adamw_torch",  # Use PyTorch's AdamW (MPS compatible)
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Logging
    logging_steps=10,
    logging_dir=f"{output_dir}/logs",
    # Evaluation
    eval_steps=200,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # Performance
    dataloader_num_workers=0,  # MPS works better with 0 workers
    fp16=False,  # MPS doesn't support fp16 training yet
    bf16=False,  # MPS doesn't support bf16
    # Other
    report_to="none",  # Change to "tensorboard" if you want logging
    seed=42,
)

# 6. Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 7. Train!
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Training will run for {training_args.num_train_epochs} epochs")
print(
    f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)
print(
    f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}"
)
print("=" * 80 + "\n")

try:
    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    # 8. Save the final model
    print("Saving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\n✅ Model saved to {output_dir}/final")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(f"{output_dir}/interrupted")
    tokenizer.save_pretrained(f"{output_dir}/interrupted")
    print(f"💾 Checkpoint saved to {output_dir}/interrupted")

except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback

    traceback.print_exc()

# 9. Evaluation
print("\n" + "=" * 80)
print("RUNNING FINAL EVALUATION")
print("=" * 80)
eval_results = trainer.evaluate()
print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("ALL DONE!")
print("=" * 80)
