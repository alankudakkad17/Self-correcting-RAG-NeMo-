"""Download the pinned HotpotQA distractor validation file into data/raw/hotpotqa/.

Usage: python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402  (sets HF_HOME before huggingface_hub is imported)

from huggingface_hub import hf_hub_download  # noqa: E402


def main():
    parquet = hf_hub_download(repo_id=paths.HOTPOT_REPO,
                              filename=paths.HOTPOT_FILE,
                              repo_type="dataset",
                              revision=paths.HOTPOT_REVISION,
                              local_dir=paths.RAW_DIR / "hotpotqa")
    print("dataset:", parquet)


if __name__ == "__main__":
    main()
