"""Persistent cache for the LLM calls the toolkit's ReAct agent makes through its LangChain client.

Our own agents call models through src/llm.py, which caches on disk. The toolkit's react_agent calls its model
through LangChain (`builder.get_llm(..., LANGCHAIN)`), so we enable LangChain's global LLM cache: an SQLite file
keyed by (prompt, model and its parameters). This is the only module allowed to import LangChain
(tests/test_no_other_orchestration.py): configuration of a library cache, not orchestration.
"""

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from src import paths

paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=str(paths.CACHE_DIR / "agent_llm_cache.sqlite")))
