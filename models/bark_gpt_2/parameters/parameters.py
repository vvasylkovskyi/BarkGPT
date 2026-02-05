from dataclasses import dataclass
import torch
from local_datasets.load_dataset_small import dataset
from logger.logger import Logger
from models.bark_gpt_2.tokenization_manager.tokenization_manager import (
    TokenizationManager,
)

logger = Logger("parameters")


@dataclass
class GPTConfig:
    vocab_size: int
    n_ctx: int  # was max_seq_len
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


class ParametersManager:
    model_config: GPTConfig
    training_parameters: TrainingParameters
    generation_parameters: GenerationParameters

    def __init__(
        self,
        model_config: GPTConfig,
        training_parameters: TrainingParameters,
        generation_parameters: GenerationParameters,
    ):
        self.model_config = model_config
        self.training_parameters = training_parameters
        self.generation_parameters = generation_parameters


### Model parameters ###
n_ctx = 128  # Context window
n_layer = 4
n_head = 2
n_embd = 256

### Training parameters ###
batch_size = 16
accum_steps = 1
effective_batch = batch_size * accum_steps  # 64 x 32 = 2048 effective batch
lr_small = 1e-4  # Original learning rate for batch_size=16
lr_scaled = lr_small * (
    effective_batch / batch_size
)  # Scale LR linearly with effective batch
epochs = 1
checkpoint_interval = 500  # Save checkpoint every N steps

### Generation parameters ###
max_length = 100
temperature = 0.8
top_k = 40

### Device setup ###
device: str = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Using device: {device}")

tokenization_manager = TokenizationManager(dataset, device, n_ctx)

vocab_size = tokenization_manager.tokenizer.vocab_size

model_config = GPTConfig(
    vocab_size=vocab_size,  # real BPE vocab
    n_ctx=n_ctx,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
)

logger.info(
    f"model parameters: vocab_size={vocab_size}, n_ctx={n_ctx}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}"
)


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

generation_parameters = GenerationParameters(
    max_length=max_length, temperature=temperature, top_k=top_k
)
logger.info(f"Generation parameters: {generation_parameters}")


parameters_manager = ParametersManager(
    model_config=model_config,
    training_parameters=training_parameters,
    generation_parameters=generation_parameters,
)
