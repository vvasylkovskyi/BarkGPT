from huggingface_hub import HfApi
from models.bark_gpt_2.constants.constants import BARK_GPT_2_PATH_IN_HUGGING_FACES_REPO, HUGGING_FACES_REPO, MODEL_PATH
api = HfApi()

api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=BARK_GPT_2_PATH_IN_HUGGING_FACES_REPO,
    repo_id=HUGGING_FACES_REPO,
)