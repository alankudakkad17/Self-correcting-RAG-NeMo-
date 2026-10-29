"""System H: the generator alone, no retrieval."""

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient
from src.prompts import LLM_ONLY_SYSTEM
from src.prompts import clean_answer
from src.state import SystemResult
from src.state import system_result_to_str


class LLMOnlyConfig(FunctionBaseConfig, name="llm_only_workflow"):
    """Answer from the model's own knowledge."""
    llm_name: LLMRef
    system_id: str = Field(default="H")


@register_function(config_type=LLMOnlyConfig)
async def llm_only_workflow(config: LLMOnlyConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))

    async def _answer(question: str) -> SystemResult:
        """Answer the question without retrieval."""
        result = await client.chat([{"role": "system", "content": LLM_ONLY_SYSTEM},
                                    {"role": "user", "content": question}])
        answer, abstained = clean_answer(result.text)
        return SystemResult(system=config.system_id, answer=answer, abstained=abstained)

    yield FunctionInfo.from_fn(_answer, description=_answer.__doc__, converters=[system_result_to_str])
