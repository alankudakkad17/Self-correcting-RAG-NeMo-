"""Search tools for the ReAct Researcher (registered toolkit functions; the agent picks them by name).

- corpus_search_tool: the frozen encyclopedia corpus (hybrid retrieval + rerank)
- web_search_tool:    the live web (Tavily)
Every passage goes through the retrieval rail (NeMo Guardrails) before the agent sees it: web and corpus text
is untrusted. Each tool also records what it returned, so the Controller can add those passages to the evidence
pool and the Verifier can check claims that cite them (see collect_tool_results).
"""

from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import register_function

from src.functions.guardrails import GuardrailRequest
from src.functions.guardrails import GuardrailResponse
from src.functions.retriever import RetrievalOutput
from src.functions.retriever import RetrievalRequest
from src.state import RetrievedDoc
from src.websearch import TavilyClient
from src.websearch import web_doc


class ToolResults:
    """What the agent's tools returned during one Researcher call."""

    def __init__(self):
        self.docs: dict[str, RetrievedDoc] = {}
        self.searches: list[str] = []

    def record(self, kind: str, query: str, docs: list[RetrievedDoc]) -> None:
        self.searches.append(f"{kind}: {query}")
        for d in docs:
            self.docs.setdefault(d.doc_id, d)


_current: ContextVar[ToolResults | None] = ContextVar("tool_results", default=None)


@contextmanager
def collect_tool_results():
    """Collect tool results for the duration of one agent run (async-safe: one collector per context)."""
    results = ToolResults()
    token = _current.set(results)
    try:
        yield results
    finally:
        _current.reset(token)


def format_passages(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "No results."
    return "\n\n".join(f"[{d.doc_id}] {d.title}{' (' + d.url + ')' if d.url else ''}: {d.text}" for d in docs)


async def sanitize(rails, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Retrieval rail over passages (no-op when guardrails are not configured)."""
    if rails is None or not docs:
        return docs
    out = await rails.ainvoke(GuardrailRequest(stage="passages", question="", docs=docs), to_type=GuardrailResponse)
    return out.docs


class CorpusSearchToolConfig(FunctionBaseConfig, name="corpus_search_tool"):
    """Agent tool: search the local encyclopedia corpus."""
    retriever: FunctionRef
    guardrails: FunctionRef | None = None
    top_k: int = Field(default=4, ge=1, le=10)


@register_function(config_type=CorpusSearchToolConfig)
async def corpus_search_tool(config: CorpusSearchToolConfig, builder: Builder):
    retriever = await builder.get_function(config.retriever)
    rails = await builder.get_function(config.guardrails) if config.guardrails else None

    async def _search(query: str) -> str:
        """Search the encyclopedia corpus. Input: a short search query. Returns passages, each with an id in
        square brackets that you can cite."""
        out = await retriever.ainvoke(RetrievalRequest(query=query, top_k=config.top_k), to_type=RetrievalOutput)
        docs = await sanitize(rails, out.docs)
        if (results := _current.get()) is not None:
            results.record("corpus", query, docs)
        return format_passages(docs)

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)


class WebSearchToolConfig(FunctionBaseConfig, name="web_search_tool"):
    """Agent tool: search the live web with Tavily."""
    guardrails: FunctionRef | None = None
    max_results: int = Field(default=3, ge=1, le=10)
    cache: bool = True


@register_function(config_type=WebSearchToolConfig)
async def web_search_tool(config: WebSearchToolConfig, builder: Builder):
    search = TavilyClient(max_results=config.max_results, cache=config.cache)
    rails = await builder.get_function(config.guardrails) if config.guardrails else None

    async def _search(query: str) -> str:
        """Search the live web. Use it only when the corpus has no answer. Input: a short search query. Returns
        web snippets, each with an id (web_...) in square brackets that you can cite, and its URL."""
        try:
            results = await search.search(query)
        except Exception as e:  # the agent sees the failure and can carry on with the corpus
            return f"Web search unavailable: {type(e).__name__}: {e}"[:300]
        docs = await sanitize(rails, [web_doc(r, i) for i, r in enumerate(results, start=1)])
        if (collected := _current.get()) is not None:
            collected.record("web", query, docs)
        return format_passages(docs)

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)
