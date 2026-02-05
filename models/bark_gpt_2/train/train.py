from models.bark_gpt_2.parameters.parameters import (
    training_parameters,
    model_config,
    device,
)
from local_datasets.load_dataset_small import dataset

from models.bark_gpt_2.tokenization_manager.tokenization_manager import (
    TokenizationManager,
)
from models.bark_gpt_2.model_checkpoint_manager.model_checkpoints_manager import (
    ModelCheckpointsManager,
)
from models.bark_gpt_2.train.training_manager import TrainingManager
from models.bark_gpt_2.train.training_debug_info import print_debug_info
from logger.logger import Logger

logger = Logger("train")

n_ctx = model_config.n_ctx

tokenization_manager = TokenizationManager(dataset, device, n_ctx)
model_checkpoints_manager = ModelCheckpointsManager(
    device, model_config, training_parameters.checkpoint_interval
)

trainer = TrainingManager(tokenization_manager, model_checkpoints_manager)
print_debug_info()
trainer.train()

logger.success("Training complete.")

model_checkpoints_manager.save_final_model_weights(trainer.model)
tokenization_manager.save_tokenizer()
