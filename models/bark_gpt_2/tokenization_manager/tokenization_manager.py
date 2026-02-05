import json
import os
from typing import Any, Dict, Sequence, cast
import time

import torch
from datasets import load_from_disk
from logger.logger import Logger
from models.bark_gpt_2.constants.constants import (
    META_TOKENIZED_CACHE,
    TOKENIZED_CACHE,
    CACHE_DIR,
    TOKENIZER_PATH,
)
from models.bark_gpt_2.ui.progress_bar import progress_bar
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

logger = Logger("TokenizationManager")


class TokenizationManager:
    device: str
    tokenizer: Any  # TODO: specify type
    n_ctx: int
    dataset: DatasetDict

    def __init__(self, dataset: Dataset | DatasetDict, device: str, n_ctx: int):
        # Normalize dataset to DatasetDict with "train" split for consistency
        if isinstance(dataset, Dataset):
            self.dataset = DatasetDict({"train": dataset})
        else:
            self.dataset = dataset

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # important
        self.n_ctx = n_ctx

    def collate_fn(self, batch: list[Dataset]) -> Dict[str, torch.Tensor]:
        # batch is a list of dicts: [{"input_ids": [...]}, ...]
        input_ids = [
            torch.tensor(item["input_ids"], device=self.device) for item in batch
        ]
        # pad sequences to the max length in this batch
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return {"input_ids": input_ids}

    def _group_texts(self, examples: Dataset) -> Dict[str, Sequence[Sequence[int]]]:
        # Flatten all token IDs
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // self.n_ctx) * self.n_ctx
        all_ids = all_ids[:total_len]

        # Split into blocks of `n_ctx`
        input_ids: Sequence[Sequence[int]] = [
            all_ids[i : i + self.n_ctx] for i in range(0, total_len, self.n_ctx)
        ]
        return {"input_ids": input_ids}

    def _load_tokenized_dataset_from_cache(self) -> Dataset:
        """
        Load the tokenized dataset from disk cache if available.
        Otherwise, tokenize and save to disk.

        Returns:
            tokenized_dataset: HuggingFace Dataset object
        """

        if os.path.exists(TOKENIZED_CACHE):
            tokenized_dataset: Dataset = cast(Dataset, load_from_disk(TOKENIZED_CACHE))

            logger.success("Loaded tokenized dataset from disk cache")
        else:
            tokenized_dataset: Dataset = self._get_dataset_training_split().map(
                self.tokenize, batched=True, remove_columns=["text"]
            )
            tokenized_dataset.save_to_disk(TOKENIZED_CACHE)
            logger.info("Tokenized dataset saved to disk cache")

        return tokenized_dataset

    def _load_grouped_dataset_from_cache(self) -> Dataset:
        if os.path.exists(CACHE_DIR):
            logger.success("Loading tokenized + grouped dataset from disk")
            lm_dataset = cast(Dataset, load_from_disk(CACHE_DIR))

        else:
            logger.info("Building tokenized + grouped dataset (one-time)")
            tokenized_dataset = self._load_tokenized_dataset_from_cache()
            lm_dataset = tokenized_dataset.map(
                self._group_texts,
                batched=True,
            )

            lm_dataset.save_to_disk(CACHE_DIR)
            logger.info("Dataset saved to disk")

        return lm_dataset

    def load_final_dataset_split_for_training(self) -> Dataset:
        return self._load_grouped_dataset_from_cache()

    def _get_dataset_training_split(self) -> Dataset:
        """Safe access: check if dataset has splits or use it directly"""
        return self.dataset["train"]

    def tokenize(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a batch of examples.
        Args:
            batch: A dictionary with keys like 'text'.
        Returns:
            A dictionary with tokenized outputs, e.g., {'input_ids': [...]}
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            return_attention_mask=False,
        )

    def get_total_tokens(self):
        """Count total tokens in the dataset"""
        total_tokens = 0
        start_time = time.time()
        total_tokens = 0
        dataset = self._load_grouped_dataset_from_cache()

        number_of_examples = len(dataset)

        input_ids: Sequence[torch.Tensor] = dataset["input_ids"]

        if os.path.exists(META_TOKENIZED_CACHE):
            with open(META_TOKENIZED_CACHE) as f:
                total_tokens = json.load(f)["total_tokens"]
        else:
            total = len(input_ids)
            for i, x in enumerate(input_ids):
                total_tokens += len(x)
                if i % 1000 == 0 or i + 1 == total:
                    progress_bar(
                        i,
                        total=total,
                        start_time=start_time,
                        prefix="Counting tokens",
                    )
            os.makedirs(os.path.dirname(META_TOKENIZED_CACHE), exist_ok=True)
            with open(META_TOKENIZED_CACHE, "w") as f:
                json.dump({"total_tokens": total_tokens}, f)

        avg_tokens_per_example = total_tokens / number_of_examples
        return total_tokens, number_of_examples, avg_tokens_per_example

    def save_tokenizer(self):
        """Save the tokenizer to disk"""
        self.tokenizer.save_pretrained(TOKENIZER_PATH)
        logger.success(f"Final tokenizer saved to {TOKENIZER_PATH}")
