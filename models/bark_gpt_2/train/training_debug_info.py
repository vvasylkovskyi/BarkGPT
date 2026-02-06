import math
from logger.logger import Logger

from models.bark_gpt_2.parameters.parameters import parameters_manager
from local_datasets.load_dataset_small import dataset
from models.bark_gpt_2.tokenization_manager.tokenization_manager import (
    TokenizationManager,
)
from models.bark_gpt_2.parameters.parameters import device

logger = Logger("debug_info")


def print_debug_info():
    logger.info("=== Training Debug Info ===")
    # -----------------------------
    # 3. Model parameters
    # -----------------------------

    # Transformer params (attention + MLP)
    transformer_params = (
        12
        * parameters_manager.model_config.n_embd**2
        * parameters_manager.model_config.n_layer
    )
    # Token embeddings
    token_emb_params = (
        parameters_manager.model_config.vocab_size
        * parameters_manager.model_config.n_embd
    )

    # Positional embeddings
    pos_emb_params = (
        parameters_manager.model_config.n_ctx * parameters_manager.model_config.n_embd
    )

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
    logger.info(
        f"Estimated model parameters: {model_params:,} (~{model_params/1e6:.2f}M)"
    )

    # -----------------------------
    # 4. Count total tokens
    # -----------------------------
    total_tokens = 0

    tokenization_manager = TokenizationManager(
        dataset, device, parameters_manager.model_config.n_ctx
    )
    total_tokens, number_of_examples, avg_tokens_per_example = (
        tokenization_manager.get_total_tokens()
    )

    logger.info(f"Total tokens: {total_tokens:,}")

    logger.info(f"Number of examples: {number_of_examples:,}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Average tokens per example: {avg_tokens_per_example:.2f}")

    # -----------------------------
    # 5. Compute number of blocks
    # -----------------------------
    n_ctx = parameters_manager.model_config.n_ctx
    num_blocks = total_tokens // n_ctx
    avg_tokens_per_block = n_ctx  # by design each block has n_ctx tokens

    logger.info(f"Number of blocks (n_ctx={n_ctx}): {num_blocks:,}")
    logger.info(f"Average tokens per block: {avg_tokens_per_block:,}")

    # -----------------------------
    # 6. Compute steps per epoch
    # -----------------------------
    batch_size = parameters_manager.training_parameters.batch_size
    accum_steps = (
        parameters_manager.training_parameters.accum_steps
    )  # gradient accumulation

    steps_per_epoch = math.ceil(num_blocks / batch_size / accum_steps)
    tokens_per_step = batch_size * n_ctx  # tokens seen per step

    logger.info(
        f"Steps per epoch (with accumulation={accum_steps}): {steps_per_epoch:,}"
    )
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

    if parameters_manager.training_parameters.epochs < min_epochs:
        logger.warning(
            f"Warning: Current epochs ({parameters_manager.training_parameters.epochs}) is less than Chinchilla minimum ({min_epochs}). Consider increasing epochs."
        )
    else:
        logger.info("Chinchilla training requirement satisfied.")
    # -----------------------------
