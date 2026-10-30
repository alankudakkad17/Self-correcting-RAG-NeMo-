"""NeMo Guardrails as a registered toolkit function. One function, four stages:

- input:      the question             -> allowed? (+ refusal message)
- passages:   retrieved passages        -> sanitized passages
- output:     the answer + claims report -> allowed? (+ final, PII-redacted answer)
- abstention: an abstention             -> safe explanation text
"""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function

from src.guardrails.rails import GuardrailSettings
from src.guardrails.rails import Guardrails
from src.state import ABSTAIN_TEXT
from src.state import GuardrailEvent
from src.state import RetrievedDoc


class GuardrailRequest(BaseModel):
    stage: Literal["input", "passages", "output", "abstention"]
    question: str
    docs: list[RetrievedDoc] = Field(default_factory=list)
    answer: str = ""
    report: dict = Field(default_factory=dict)


class GuardrailResponse(BaseModel):
    allowed: bool = True
    text: str = ""
    docs: list[RetrievedDoc] = Field(default_factory=list)
    events: list[GuardrailEvent] = Field(default_factory=list)


class GuardrailsConfig(FunctionBaseConfig, GuardrailSettings, name="guardrails"):
    """Input, retrieval, output and abstention rails (NeMo Guardrails, Colang 1.0). Each rail can be switched off."""


@register_function(config_type=GuardrailsConfig)
async def guardrails(config: GuardrailsConfig, builder: Builder):
    rails = Guardrails(GuardrailSettings(**{k: getattr(config, k) for k in GuardrailSettings.model_fields}))

    async def _apply(request: GuardrailRequest) -> GuardrailResponse:
        """Apply the rails for one stage."""
        if request.stage == "input":
            allowed, refusal, events = await rails.check_input(request.question)
            return GuardrailResponse(allowed=allowed, text=refusal, events=events)
        if request.stage == "passages":
            docs, events = await rails.sanitize(request.docs)
            return GuardrailResponse(docs=docs, events=events)
        if request.stage == "output":
            allowed, text, events = await rails.check_output(request.question, request.answer, request.report)
            return GuardrailResponse(allowed=allowed, text=text, events=events)
        text, events = await rails.abstention(request.question, request.answer or ABSTAIN_TEXT)
        return GuardrailResponse(text=text, events=events)

    yield FunctionInfo.from_fn(_apply, description=_apply.__doc__)
