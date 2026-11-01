"""Toolkit plugin entry point (see [project.entry-points."nat.components"] in pyproject.toml).

Importing a module runs its @register_* decorators, which makes its `_type` names usable in YAML.
The ReAct agent (`_type: react_agent`) comes from the toolkit's LangChain plugin (nvidia-nat-langchain).
"""

# flake8: noqa: F401
from src.functions import agentic_researcher
from src.functions import fact_checker
from src.functions import guardrails
from src.functions import planner
from src.functions import researcher
from src.functions import retriever
from src.functions import search_tools
from src.functions import verifier
from src.workflows import self_correcting
from src.workflows import smoke
