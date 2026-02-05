import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
from models.bark_gpt_2.ui.progress_bar import progress_bar
from models.bark_gpt_2.model.model import BarkGPT
from models.bark_gpt_2.parameters.parameters import (
    training_parameters,
    model_config,
    device,
    GPTConfig,
)
from datasets import Dataset

from models.bark_gpt_2.tokenization_manager.tokenization_manager import (
    TokenizationManager,
)
from models.bark_gpt_2.model_checkpoint_manager.model_checkpoints_manager import (
    ModelCheckpointsManager,
)

from logger.logger import Logger

logger = Logger("TrainingManager")


class TrainingManager:
    model_config: GPTConfig
    tokenizationManager: TokenizationManager
    modelCheckpointsManager: ModelCheckpointsManager
    train_loader: DataLoader
    loss_fn: nn.CrossEntropyLoss
    optimizer: torch.optim.Optimizer
    model: BarkGPT

    batch_size: int
    epochs: int
    accum_steps: int
    effective_batch: int
    lr_small: float
    lr_scaled: float

    def __init__(
        self,
        tokenization_manager: TokenizationManager,
        model_checkpoints_manager: ModelCheckpointsManager,
    ):
        self.batch_size = training_parameters.batch_size
        self.epochs = training_parameters.epochs
        self.accum_steps = training_parameters.accum_steps
        self.effective_batch = training_parameters.effective_batch
        self.checkpoint_interval = training_parameters.checkpoint_interval
        self.lr_small = training_parameters.lr_small
        self.lr_scaled = training_parameters.lr_scaled

        self.tokenization_manager = tokenization_manager

        self.model_checkpoints_manager = model_checkpoints_manager

        self.prepare_training()

    def prepare_training(
        self,
    ):
        lm_dataset: Dataset = (
            self.tokenization_manager.load_final_dataset_split_for_training()
        )

        self.train_loader = DataLoader(
            lm_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.tokenization_manager.collate_fn,
        )

        self.model = BarkGPT(model_config).to(device)
        self.loss_fn = nn.CrossEntropyLoss()  # next-token prediction
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_small)

        (
            self.start_epoch,
            self.start_step,
            self.global_step,
            model_chkp,
            optimizer_ckpt,
        ) = self.model_checkpoints_manager.load_checkpoint()

        if model_chkp is not None:
            self.model.load_state_dict(model_chkp)
        if optimizer_ckpt is not None:
            self.optimizer.load_state_dict(optimizer_ckpt)

    def sanity_loss(self, sample_batch: torch.Tensor):
        self.model.eval()  # disable dropout / training effects
        with torch.no_grad():  # do NOT compute gradients
            inputs = sample_batch[:, :-1]
            targets = sample_batch[:, 1:]
            logits = self.model(inputs)
            loss = self.loss_fn(
                logits.reshape(-1, model_config.vocab_size), targets.reshape(-1)
            ).item()
        self.model.train()  # restore training mode
        return loss

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            total_loss = 0
            start_time = time.time()
            for step, batch in enumerate(self.train_loader):
                # skip until we reach the saved step
                if epoch == self.start_epoch and step < self.start_step:
                    continue

                # Minimal progress output
                progress_bar(
                    step,
                    total=len(self.train_loader),
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
                logits = self.model(inputs)
                loss = self.loss_fn(
                    logits.reshape(-1, model_config.vocab_size),
                    targets.reshape(-1),
                )
                loss = loss / self.accum_steps  # scale for accumulation
                loss.backward()

                # Early NaN/Inf detection
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"NaN/Inf detected at epoch {epoch}, step {step}, global_step {self.global_step}"
                    )
                    return  # abort training

                # # Optional gradient clipping for stability
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                total_loss += loss.item() * self.accum_steps  # scale back for logging

                # -------------------------
                # Step optimizer every accum_steps
                # -------------------------
                if (step + 1) % self.accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    if (
                        self.global_step
                        % self.model_checkpoints_manager.checkpoint_interval
                        == 0
                    ):
                        ckpt_loss = self.sanity_loss(input_ids)  # small batch
                        logger.info(
                            f"Checkpoint at step {self.global_step}: sanity loss={ckpt_loss:.4f}"
                        )
                        self.model_checkpoints_manager.save_checkpoint(
                            epoch, step, self.global_step, self.model, self.optimizer
                        )

            logger.success(
                f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}"
            )
            self.model_checkpoints_manager.save_checkpoint(
                epoch + 1, 0, self.global_step, self.model, self.optimizer
            )
