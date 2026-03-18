# sft_lora.py
import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from models.bark_gpt_2.parameters.parameters import (
    training_parameters,
    model_config,
    device,
)
# from local_datasets.load_dataset import dataset  # replace with your SFT dataset

from models.bark_gpt_2.tokenization_manager.tokenization_manager import (
    AutoTokenizer,
    TokenizationManager,
)
# from models.bark_gpt_2.model_checkpoint_manager.model_checkpoints_manager import (
#     ModelCheckpointsManager,
# )
from models.bark_gpt_2.model.model import BarkGPT
from models.bark_gpt_2.model.hf.bark_hf import BarkHF, BarkConfig

# from models.bark_gpt_2.train.training_manager import TrainingManager
# from models.bark_gpt_2.train.training_debug_info import print_debug_info
from logger.logger import Logger
from datasets import load_dataset


logger = Logger("SFT-LoRA")

# ---------------------------
# LoRA configuration
# ---------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ---------------------------
# Dataset & tokenization
# ---------------------------
n_ctx = model_config.n_ctx
# tokenization_manager = TokenizationManager(dataset, device, n_ctx)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # important
# ---------------------------
# Model + LoRA
# ---------------------------
# model = AutoModelForCausalLM.from_pretrained("vvasylkovskyi/barkgpt")

# Create the base BarkGPT model
bark_model = BarkGPT(model_config).to(device)

# Wrap it in a PreTrainedModel for HuggingFace compatibility
hf_config = BarkConfig(n_layer=model_config.n_layer)
pretrained_model = BarkHF(hf_config, bark_model)

# Apply LoRA adapters
model: PeftModel | PeftMixedModel = get_peft_model(pretrained_model, lora_config)
model.print_trainable_parameters()  # optional, check LoRA adapters

# Setup chat template for the tokenizer
if tokenizer.chat_template is None:
    print(">>> has chat_template")
    # Use a standard ChatML template
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # # Add special tokens if they don't exist
    # special_tokens = {
    #     "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    # }
    # num_added = tokenizer.add_special_tokens(special_tokens)

    # # Resize model token embeddings to match tokenizer if new tokens were added
    # if num_added > 0:
    #     # Access the base model through PEFT wrapper and resize embeddings
    #     base_model = model.get_base_model()
    #     base_model.resize_token_embeddings(len(tokenizer))
    #     logger.info(f"Resized token embeddings to accommodate {num_added} new tokens")

# TokenizationManager.tokenizer.chat_format
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
# # ---------------------------
# # Checkpoints manager
# # ---------------------------
# model_checkpoints_manager = ModelCheckpointsManager(
#     device, model_config, training_parameters.checkpoint_interval
# )

# # ---------------------------
# # Training manager
# # ---------------------------
# trainer = TrainingManager(tokenization_manager, model_checkpoints_manager)

# Inject LoRA model into trainer
# trainer.model = model
# trainer.optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, trainer.model.parameters()),
#     lr=training_parameters.lr_small,
# )

# print_debug_info()

# ---------------------------
# Start training
# ---------------------------
# trainer.train()

logger.success("LoRA SFT Training complete.")

# ---------------------------
# Save model and tokenizer
# ---------------------------
# Save full model with LoRA adapters
# model_checkpoints_manager.save_final_model_weights(trainer.model)
# tokenization_manager.save_tokenizer()

# # Optional: save LoRA adapters separately
# trainer.model.save_pretrained("barkgpt_lora")
# logger.success("LoRA adapters saved to barkgpt_lora/")