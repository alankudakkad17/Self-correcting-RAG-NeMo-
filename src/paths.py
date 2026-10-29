"""Project paths in one place. Everything (data, models, caches) stays inside the project folder."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
EVAL_DIR = ROOT / "data" / "eval"
INDEX_DIR = ROOT / "data" / "index"
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = CACHE_DIR / "models"
RESULTS_DIR = ROOT / "eval" / "results"

# Keep Hugging Face downloads out of the user profile.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))

# Pinned sources (exact revisions, so a re-download gives identical bytes)
HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_FILE = "distractor/validation-00000-of-00001.parquet"

RERANKER_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RERANKER_DIR = MODELS_DIR / "ms-marco-MiniLM-L6-v2"
