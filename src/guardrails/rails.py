"""Runs the Colang rails in this folder through NeMo Guardrails, stage by stage.

Guardrails never generates text here (no main LLM, dialog rails off): each call runs one stage's rail flows on
the given text and we read back the (possibly rewritten) text and which flow, if any, stopped it.
Every rail can be switched off in GuardrailSettings (configs/system.yml) for ablations.

Each stage gets its own LLMRails instance, built from only that stage's .co file and enabled flows. In 0.24.1
(Colang 1.0) loading all flows into one config made flows of other stages fire too (an assignment-only flow such
as the abstention message ran on every output call), and flow names in the generation options did not
restrict which output rails ran.
"""

import json
from pathlib import Path

from nemoguardrails import LLMRails
from nemoguardrails import RailsConfig
from pydantic import BaseModel

from src.guardrails import actions
from src.state import GuardrailEvent
from src.state import RetrievedDoc

CONFIG_DIR = Path(__file__).resolve().parent
# stage -> (Colang file, Guardrails rail kind it runs as)
STAGES = {"input": ("input.co", "input"), "passages": ("retrieval.co", "output"),
          "output": ("output.co", "output"), "abstention": ("abstention.co", "output")}
ACTIONS = [actions.check_question_injection, actions.check_question_topic, actions.check_question_safety,
           actions.sanitize_passage_text, actions.check_answer_grounded, actions.check_citation_format,
           actions.redact_answer_pii]

INPUT_FLOWS = {"injection": "check prompt injection", "off_topic": "check off topic", "unsafe": "check unsafe request"}
OUTPUT_FLOWS = {"groundedness": "check groundedness", "citations": "check citation format", "pii": "redact pii"}
PASSAGE_FLOW = "sanitize passage"
ABSTENTION_FLOW = "abstention response"

REFUSALS = {
    "check prompt injection": "I can't process this question: it contains instructions aimed at the assistant.",
    "check off topic": "I only answer factual questions that the document collection can answer.",
    "check unsafe request": "I can't help with that request.",
}


class GuardrailSettings(BaseModel):
    injection: bool = True  # input: block prompt injection in the question
    off_topic: bool = True  # input: block requests that are not factual questions
    unsafe: bool = True  # input: block requests for help causing harm
    retrieval: bool = True  # retrieval: strip instruction-like sentences from passages
    groundedness: bool = True  # output: answer must be accepted and backed by supported claims
    citations: bool = True  # output: every claim cites a known passage with a quote
    pii: bool = True  # output: redact emails, phone, card and ID numbers
    abstention: bool = True  # abstention: replace the bare abstention with a safe explanation


class Guardrails:

    def __init__(self, settings: GuardrailSettings):
        self.settings = settings
        self._rails: dict[tuple, LLMRails] = {}  # (stage, flows) -> LLMRails running exactly those flows
        self._passages: dict[str, tuple[str, bool]] = {}  # doc_id -> (sanitized text, changed)

    def _rails_for(self, stage: str, flows: list[str]) -> LLMRails:
        key = (stage, tuple(flows))
        if key not in self._rails:
            colang_file, kind = STAGES[stage]
            yaml = f'colang_version: "1.0"\nmodels: []\nrails:\n  {kind}:\n    flows:\n'
            yaml += "".join(f"      - {f}\n" for f in flows)
            config = RailsConfig.from_content(colang_content=(CONFIG_DIR / colang_file).read_text(encoding="utf-8"),
                                              yaml_content=yaml)
            rails = LLMRails(config)
            for fn in ACTIONS:
                rails.register_action(fn, name=fn.action_meta["name"])
            self._rails[key] = rails
        return self._rails[key]

    async def _run(self, stage: str, flows: list[str], messages: list[dict]) -> tuple[str, str | None]:
        """Run the given flows of one stage. Returns (final text, name of the flow that stopped it or None)."""
        kind = STAGES[stage][1]
        res = await self._rails_for(stage, flows).generate_async(
            messages=messages, options={"rails": [kind], "log": {"activated_rails": True}})
        text = res.response[-1]["content"] if res.response else ""
        stopped = next((r.name for r in (res.log.activated_rails if res.log else []) if r.stop), None)
        return text, stopped

    async def check_input(self, question: str) -> tuple[bool, str, list[GuardrailEvent]]:
        """(allowed, refusal message if blocked, events)."""
        flows = [f for key, f in INPUT_FLOWS.items() if getattr(self.settings, key)]
        if not flows:
            return True, "", []
        _, stopped = await self._run("input", flows, [{"role": "user", "content": question}])
        if stopped:
            return False, REFUSALS.get(stopped, "I can't help with that request."), [
                GuardrailEvent(rail="input", name=stopped, action="blocked")]
        return True, "", [GuardrailEvent(rail="input", name=", ".join(flows), action="passed")]

    async def sanitize(self, docs: list[RetrievedDoc]) -> tuple[list[RetrievedDoc], list[GuardrailEvent]]:
        """Treat passages as untrusted: remove instruction-like sentences before any model sees them."""
        if not self.settings.retrieval:
            return docs, []
        out, events = [], []
        for doc in docs:
            if doc.doc_id not in self._passages:
                text, _ = await self._run("passages", [PASSAGE_FLOW], [{"role": "user", "content": "passage"},
                                                                       {"role": "assistant", "content": doc.text}])
                self._passages[doc.doc_id] = (text, text != doc.text)
            text, changed = self._passages[doc.doc_id]
            if changed:
                events.append(GuardrailEvent(rail="retrieval", name=PASSAGE_FLOW, action="modified",
                                             detail=f"removed instruction-like text from {doc.doc_id}"))
                doc = doc.model_copy(update={"text": text})
            out.append(doc)
        return out, events

    async def check_output(self, question: str, answer: str, report: dict) -> tuple[bool, str, list[GuardrailEvent]]:
        """(allowed, final answer text, events). `report` is the Controller's claims report (see actions.py)."""
        flows = [f for key, f in OUTPUT_FLOWS.items() if getattr(self.settings, key)]
        if not flows:
            return True, answer, []
        text, stopped = await self._run("output", flows, [{"role": "context", "content": {"report": json.dumps(report)}},
                                                          {"role": "user", "content": question},
                                                          {"role": "assistant", "content": answer}])
        if stopped:
            return False, answer, [GuardrailEvent(rail="output", name=stopped, action="blocked")]
        action = "modified" if text != answer else "passed"
        detail = "redacted personal data" if action == "modified" else ""
        return True, text, [GuardrailEvent(rail="output", name=", ".join(flows), action=action, detail=detail)]

    async def abstention(self, question: str, fallback: str) -> tuple[str, list[GuardrailEvent]]:
        """Safe, explained abstention message (or `fallback` when the abstention rail is off)."""
        if not self.settings.abstention:
            return fallback, []
        text, _ = await self._run("abstention", [ABSTENTION_FLOW], [{"role": "user", "content": question},
                                                                    {"role": "assistant", "content": fallback}])
        return text, [GuardrailEvent(rail="abstention", name=ABSTENTION_FLOW, action="modified")]
