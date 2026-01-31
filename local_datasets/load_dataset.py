from datasets import load_dataset
from logger.logger import Logger

logger = Logger("dataset_loader")

dataset = load_dataset("Skylion007/openwebtext")

logger.info(f"Dataset loaded with {len(dataset['train'])} training examples.")
logger.info(f"Dataset Keys: {dataset.keys()}")
