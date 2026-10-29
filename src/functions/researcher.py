"""Researcher: evidence -> Draft (short answer + claims, each citing a doc id and a quoted span). One LLM call."""

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient
from src.parsing import parse_draft
from src.prompts import RESEARCHER_SYSTEM
from src.prompts import clean_answer
from src.prompts import researcher_user_prompt
from src.state import Draft
from src.state import ResearchRequest


class ResearcherConfig(FunctionBaseConfig, name="researcher"):
    """Draft a cited answer from retrieved passages."""
    llm_name: LLMRef


@register_function(config_type=ResearcherConfig)
async def researcher(config: ResearcherConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))

    async def _research(request: ResearchRequest) -> Draft:
        """Draft an answer with claim-level citations."""
        prompt = researcher_user_prompt(request.question, request.evidence, request.sub_questions, request.feedback)
        result = await client.chat([{"role": "system", "content": RESEARCHER_SYSTEM},
                                    {"role": "user", "content": prompt}])
        draft = parse_draft(result.text, {d.doc_id for d in request.evidence})
        draft.answer, _ = clean_answer(draft.answer)
        return draft

    yield FunctionInfo.from_fn(_research, description=_research.__doc__)
