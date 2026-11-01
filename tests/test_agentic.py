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
