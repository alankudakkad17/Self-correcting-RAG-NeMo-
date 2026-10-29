"""Toolkit plugin entry point (see [project.entry-points."nat.components"] in pyproject.toml).

Importing a module runs its @register_* decorators, which makes its `_type` names usable in YAML.
"""

# flake8: noqa: F401
from src.functions import planner
from src.functions import researcher
from src.functions import retriever
from src.functions import verifier
from src.workflows import baseline
from src.workflows import llm_only
from src.workflows import self_correcting
from src.workflows import smoke
