"""Planner: question -> Plan (1-4 sub-questions). One LLM call."""

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient
from src.parsing import parse_plan
from src.prompts import PLANNER_SYSTEM
from src.prompts import planner_user_prompt
from src.state import Plan


class PlannerConfig(FunctionBaseConfig, name="planner"):
    """Decompose a multi-hop question into sub-questions."""
    llm_name: LLMRef


@register_function(config_type=PlannerConfig)
async def planner(config: PlannerConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))

    async def _plan(question: str) -> Plan:
        """Break the question into 1-4 sub-questions."""
        result = await client.chat([{"role": "system", "content": PLANNER_SYSTEM},
                                    {"role": "user", "content": planner_user_prompt(question)}])
        return parse_plan(result.text, question)

    yield FunctionInfo.from_fn(_plan, description=_plan.__doc__)
