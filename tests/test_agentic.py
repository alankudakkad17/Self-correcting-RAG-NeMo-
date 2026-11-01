"""ReAct Researcher plumbing: tool-result collection, web passages, final-answer extraction. No LLM or network."""

import asyncio

from src.functions.agentic_researcher import final_answer_text
from src.functions.search_tools import collect_tool_results
from src.functions.search_tools import format_passages
from src.functions.search_tools import web_doc
from src.prompts import agent_research_prompt
from src.state import RetrievedDoc
from src.functions import search_tools


def test_web_doc_ids_are_stable_and_distinct():
    a = web_doc({"title": "T", "url": "https://a", "content": "text", "score": 0.5}, 1)
    assert a.doc_id == web_doc({"title": "T", "url": "https://a", "content": "text"}, 2).doc_id
    assert a.doc_id != web_doc({"title": "T", "url": "https://b", "content": "text"}, 1).doc_id
    assert a.doc_id.startswith("web_") and a.url == "https://a"


def test_format_passages_shows_citable_ids_and_urls():
    docs = [RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="Film maker.", score=1, rank=1),
            RetrievedDoc(doc_id="web_1", title="Page", text="Snippet.", score=1, rank=2, url="https://p")]
    assert format_passages(docs) == "[Ed_Wood] Ed Wood: Film maker.\n\n[web_1] Page (https://p): Snippet."
    assert format_passages([]) == "No results."


async def test_collector_is_per_run_and_concurrent_runs_do_not_mix():
    async def fake_tool(query):
        doc = RetrievedDoc(doc_id=query, title=query, text=query, score=1, rank=1)
        await asyncio.sleep(0)
        search_tools._current.get().record("corpus", query, [doc, doc])  # duplicates collapse

    async def run(query):
        with collect_tool_results() as found:
            await fake_tool(query)
            return found

    a, b = await asyncio.gather(run("alpha"), run("beta"))
    assert list(a.docs) == ["alpha"] and list(b.docs) == ["beta"]
    assert a.searches == ["corpus: alpha"]
    assert search_tools._current.get() is None  # reset after the run


def test_final_answer_text():
    assert final_answer_text('Thought: done\nFinal Answer: {"answer": "x"}') == ' {"answer": "x"}'
    assert final_answer_text('{"answer": "x"}') == '{"answer": "x"}'  # native tool calling: no marker


def test_agent_prompt_names_the_tools_and_includes_passages():
    doc = RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="Film maker.", score=1, rank=1)
    prompt = agent_research_prompt("Who was Ed Wood?", [doc], ["Who is Ed Wood?"], "")
    assert "corpus_search" in prompt and "web_search" in prompt
    assert "[Ed_Wood] Ed Wood: Film maker." in prompt and "Question: Who was Ed Wood?" in prompt


def test_corpus_only_prompt_never_mentions_web_search():
    doc = RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="Film maker.", score=1, rank=1)
    prompt = agent_research_prompt("Who won the 2022 World Cup?", [doc], web_search=False)
    assert "web_search" not in prompt and "no web access" in prompt
    assert "web_search" in agent_research_prompt("Q?", [doc], web_search=True)


class FakeBuilder:
    """Minimal stand-in for the toolkit builder's get_function_config."""

    def __init__(self, configs):
        self.configs = configs

    def get_function_config(self, name):
        if name not in self.configs:
            raise ValueError(f"unknown function {name}")
        return self.configs[name]


def test_web_search_is_detected_from_the_agent_tool_list():
    from types import SimpleNamespace

    from src.functions.agentic_researcher import has_tool
    from src.functions.search_tools import CorpusSearchToolConfig
    from src.functions.search_tools import WebSearchToolConfig

    tools = {"corpus_search": CorpusSearchToolConfig(retriever="retriever"), "web_search": WebSearchToolConfig()}
    with_web = FakeBuilder({**tools, "agent": SimpleNamespace(tool_names=["corpus_search", "web_search"])})
    without = FakeBuilder({**tools, "agent": SimpleNamespace(tool_names=["corpus_search", "some_group"])})
    assert has_tool(with_web, "agent", WebSearchToolConfig) is True
    assert has_tool(without, "agent", WebSearchToolConfig) is False  # unknown names (e.g. groups) are skipped


def test_docs_search_is_mentioned_only_when_available():
    doc = RetrievedDoc(doc_id="Ed_Wood", title="Ed Wood", text="Film maker.", score=1, rank=1)
    assert "docs_search" in agent_research_prompt("What does my report say?", [doc], docs_search=True)
    assert "docs_search" not in agent_research_prompt("Q?", [doc], docs_search=False)


def test_only_relevant_unheld_upload_chunks_are_kept():
    from src.functions.search_tools import relevant_uploads

    def chunk(i, score):
        return RetrievedDoc(doc_id=f"up_x_{i}", title="memo.md", text="t", score=score, rank=i)

    docs = [chunk(0, 0.55), chunk(1, 0.46), chunk(2, 0.33), chunk(3, 0.03)]
    assert [d.doc_id for d in relevant_uploads(docs, 0.2, exclude={"up_x_1"}, k=5)] == ["up_x_0", "up_x_2"]
    assert len(relevant_uploads(docs, 0.2, exclude=set(), k=2)) == 2  # top_k respected
    assert relevant_uploads([chunk(0, 0.03)], 0.2, exclude=set(), k=3) == []  # unrelated question: nothing
