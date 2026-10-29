"""Systems A and B: standard RAG. One retrieval with the question, one generation over the top-k passages.

A and B share this workflow; they differ only in the retriever config (vector vs hybrid + rerank).
"""

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.functions.retriever import RetrievalOutput
from src.functions.retriever import RetrievalRequest
from src.llm import ChatClient
from src.prompts import RAG_SYSTEM
from src.prompts import clean_answer
from src.prompts import rag_user_prompt
from src.state import SystemResult
from src.state import system_result_to_str


class BaselineRAGConfig(FunctionBaseConfig, name="baseline_rag_workflow"):
    """Retrieve once, generate once."""
    llm_name: LLMRef
    retriever: FunctionRef = Field(description="A `corpus_retriever` entry under `functions:`.")
    system_id: str = Field(default="A")


@register_function(config_type=BaselineRAGConfig)
async def baseline_rag_workflow(config: BaselineRAGConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))
    retriever = await builder.get_function(config.retriever)

    async def _answer(question: str) -> SystemResult:
        """Answer the question from retrieved corpus passages."""
        retrieved = await retriever.ainvoke(RetrievalRequest(query=question), to_type=RetrievalOutput)
        result = await client.chat([{"role": "system", "content": RAG_SYSTEM},
                                    {"role": "user", "content": rag_user_prompt(question, retrieved.docs)}])
        answer, abstained = clean_answer(result.text)
        return SystemResult(system=config.system_id, answer=answer, abstained=abstained, retrieved=retrieved.docs)

    yield FunctionInfo.from_fn(_answer, description=_answer.__doc__, converters=[system_result_to_str])
