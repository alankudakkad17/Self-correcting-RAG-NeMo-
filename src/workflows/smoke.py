"""Milestone 1 smoke workflow: one cached call to the configured LLM.

Run:  nat run --config_file configs/smoke.yml --input "What is the capital of France?"
"""

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient


class SmokeWorkflowConfig(FunctionBaseConfig, name="smoke_workflow"):
    """Answer a question with a single LLM call (no retrieval)."""
    llm_name: LLMRef = Field(description="Which entry under `llms:` to call.")
    system_prompt: str = Field(default="Answer in one short sentence.")


@register_function(config_type=SmokeWorkflowConfig)
async def smoke_workflow(config: SmokeWorkflowConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))

    async def _answer(question: str) -> str:
        """Answer the question in one sentence."""
        result = await client.chat([
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": question},
        ])
        return result.text

    yield FunctionInfo.from_fn(_answer, description=_answer.__doc__)
