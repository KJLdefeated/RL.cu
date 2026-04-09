# scripts/download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./Qwen3-0.6B",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)