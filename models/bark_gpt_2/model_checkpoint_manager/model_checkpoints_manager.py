import os
import torch
from models.bark_gpt_2.constants.constants import (
    CKPT_PATH,
    MODEL_PATH,
)
from logger.logger import Logger
from torch import nn
from models.bark_gpt_2.parameters.parameters import GPTConfig

logger = Logger("ModelCheckpointsManager")


class ModelCheckpointsManager:
    device: torch.device
    checkpoint_interval: int
    model_config: GPTConfig

    def __init__(
        self, device: torch.device, model_config: GPTConfig, checkpoint_interval: int
    ):
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.model_config = model_config

    def save_checkpoint(
        self,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
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
                "vocab_size": self.model_config.vocab_size,
                "max_seq_len": self.model_config.n_ctx,
            },
            tmp,
        )
        os.replace(tmp, CKPT_PATH)

    def save_final_model_weights(self, model: nn.Module):
        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": self.model_config.vocab_size,
                "max_seq_len": self.model_config.n_ctx,
            },
            MODEL_PATH,
        )
        logger.success(f"Final model saved to {MODEL_PATH}")

    def load_final_model_weights(self):
        return torch.load(MODEL_PATH, map_location=self.device)

    def load_checkpoint(
        self,
    ):
        if os.path.exists(CKPT_PATH):
            ckpt = torch.load(CKPT_PATH, map_location=self.device)
            model_chkp = ckpt["model"]
            optimizer_ckpt = ckpt["optim"]
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            global_step = ckpt.get("global_step", 0)
            logger.success(f"Resumed from epoch {start_epoch}, step {start_step}")
            return start_epoch, start_step, global_step, model_chkp, optimizer_ckpt
        else:
            logger.info("No checkpoint found, starting fresh training")
            return 0, 0, 0, None, None
