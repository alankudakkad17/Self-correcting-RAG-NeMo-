"""Download the pinned HotpotQA distractor validation file and the reranker model.

Usage: python scripts/download_data.py
Both land inside the project: data/raw/hotpotqa/ and data/cache/models/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402  (sets HF_HOME before huggingface_hub is imported)

from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def main():
    parquet = hf_hub_download(repo_id=paths.HOTPOT_REPO,
                              filename=paths.HOTPOT_FILE,
                              repo_type="dataset",
                              revision=paths.HOTPOT_REVISION,
                              local_dir=paths.RAW_DIR / "hotpotqa")
    print("dataset :", parquet)

    # Weights only (safetensors); no custom code from the model repo is ever executed.
    model_dir = snapshot_download(repo_id=paths.RERANKER_REPO,
                                  revision=paths.RERANKER_REVISION,
                                  local_dir=paths.RERANKER_DIR,
                                  allow_patterns=["*.json", "*.txt", "model.safetensors"])
    print("reranker:", model_dir)


if __name__ == "__main__":
    main()
