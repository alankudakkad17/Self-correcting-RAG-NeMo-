"""Hard constraint: orchestration uses the NeMo Agent Toolkit only.

LangChain/LangGraph are installed as dependencies of RAGAS and Guardrails, so we check our own code.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(langgraph|langchain\w*|llama_index|crewai|autogen)\b", re.MULTILINE)


def test_src_imports_no_other_orchestration_framework():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
