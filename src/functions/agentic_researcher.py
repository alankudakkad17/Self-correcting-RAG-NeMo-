"""ReAct Researcher: the toolkit's built-in `react_agent` drafts the cited answer, searching as it needs to.

The agent is configured in YAML (functions.research_agent, `_type: react_agent`) with the corpus and web search
tools; this function gives it the task, collects what its tools returned, and parses the cited JSON draft.
The Verifier and the rule-based Controller still decide accept / retry / revise / abstain, so the loop stays
bounded however the agent behaves. If the agent fails (tool-call budget used up, unparseable reply), the result
is a draft with no claims, which the Controller treats like any unsupported draft.
"""

import logging

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import register_function

from src import agent_cache  # noqa: F401  (persistent cache for the agent's LLM calls)
from src.functions.search_tools import collect_tool_results
from src.parsing import parse_draft
from src.prompts import agent_research_prompt
from src.prompts import clean_answer
from src.state import ABSTAIN_TEXT
from src.state import Draft
from src.state import ResearchOutput
from src.state import ResearchRequest

logger = logging.getLogger(__name__)
FINAL_ANSWER = "Final Answer:"


def final_answer_text(agent_output: str) -> str:
    """The part after the last "Final Answer:" marker (text-mode ReAct), else the whole reply."""
    idx = agent_output.rfind(FINAL_ANSWER)
    return agent_output[idx + len(FINAL_ANSWER):] if idx != -1 else agent_output


class AgenticResearcherConfig(FunctionBaseConfig, name="agentic_researcher"):
    """Draft a cited answer with a toolkit ReAct agent that can search the corpus and the web."""
    agent: FunctionRef  # a `react_agent` entry under functions:


@register_function(config_type=AgenticResearcherConfig)
async def agentic_researcher(config: AgenticResearcherConfig, builder: Builder):
    agent = await builder.get_function(config.agent)

    async def _research(request: ResearchRequest) -> ResearchOutput:
        """Draft an answer with claim-level citations, searching for more evidence when needed."""
        task = agent_research_prompt(request.question, request.evidence, request.sub_questions, request.feedback)
        with collect_tool_results() as found:
            try:
                reply = await agent.ainvoke(task, to_type=str)
            except Exception as e:  # budget exhausted or parsing failed: no draft, the Controller decides
                logger.warning("ReAct Researcher failed: %s: %s", type(e).__name__, e)
                reply = ""
        known = {d.doc_id for d in request.evidence}
        new_docs = [d for d in found.docs.values() if d.doc_id not in known]
        draft = parse_draft(final_answer_text(reply), known | {d.doc_id for d in new_docs}) if reply \
            else Draft(answer=ABSTAIN_TEXT)
        draft.answer, _ = clean_answer(draft.answer)
        return ResearchOutput(draft=draft, new_docs=new_docs, searches=found.searches)

    yield FunctionInfo.from_fn(_research, description=_research.__doc__)
