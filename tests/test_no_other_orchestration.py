"""Orchestration is the NeMo Agent Toolkit's: our code never builds LangChain / LangGraph chains or agents.

The toolkit's own ReAct agent (nvidia-nat-langchain) runs on LangGraph internally; we only configure it in YAML.
The single allowed LangChain import is src/agent_cache.py, which turns on LangChain's LLM cache for that agent.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(langgraph|langchain\w*|llama_index|crewai|autogen)\b", re.MULTILINE)
ALLOWED = {"agent_cache.py"}


def test_src_imports_no_other_orchestration_framework():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
                 if str(p.relative_to(SRC)) not in ALLOWED and FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_agent_cache_only_configures_the_cache():
    imports = re.findall(r"^\s*from\s+(\S+)\s+import\s+(\S+)", (SRC / "agent_cache.py").read_text(encoding="utf-8"),
                         re.MULTILINE)
    lc = {(m, n) for m, n in imports if m.startswith("langchain")}
    assert lc == {("langchain_community.cache", "SQLiteCache"), ("langchain_core.globals", "set_llm_cache")}
