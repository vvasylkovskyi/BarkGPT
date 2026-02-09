from huggingface_hub import snapshot_download

snapshot_download(
    "mistralai/Ministral-3-3B-Base-2512", cache_dir="./outputs/mistral-3b-base"
)

# snapshot_download(
#     repo_id="mistralai/Mistral-7B-v0.3",
#     cache_dir="./outputs/mistral-7b-v0.3",
# )
