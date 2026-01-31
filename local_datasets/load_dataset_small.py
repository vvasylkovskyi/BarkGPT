import os

with open(os.environ["DATASET_PATH"], "r", encoding="utf-8") as f:
    texts = f.readlines()

# Wrap as HF-style dataset
from datasets import Dataset

dataset = Dataset.from_dict({"text": [t for t in texts if t]})
print(f"Number of examples: {len(dataset)}")
