import os


MODEL_NAME = "BarkGPT-2"
TOKENIZED_CACHE = f"{os.environ['DATASETS_CACHE_PATH']}/{MODEL_NAME}/tokenized_dataset"
META_TOKENIZED_CACHE = (
    f"{os.environ['DATASETS_CACHE_PATH']}/{MODEL_NAME}/lm_dataset_meta/meta.json"
)

CKPT_PATH = f"{os.path.join(os.environ['MODEL_CHECKPOINTS_PATH'], MODEL_NAME)}"
CACHE_DIR = f"{os.environ['DATASETS_CACHE_PATH']}/{MODEL_NAME}/lm_dataset"
MODEL_PATH = f"{os.environ['MODEL_PATH']}/{MODEL_NAME}_model.pt"
TOKENIZER_PATH = f"{os.environ['MODEL_TOKENIZER_PATH']}/{MODEL_NAME}_tokenizer"

os.makedirs(os.environ["MODEL_CHECKPOINTS_PATH"], exist_ok=True)
os.makedirs(os.environ["MODEL_PATH"], exist_ok=True)
os.makedirs(os.environ["MODEL_TOKENIZER_PATH"], exist_ok=True)
os.makedirs(os.environ["DATASETS_CACHE_PATH"], exist_ok=True)
