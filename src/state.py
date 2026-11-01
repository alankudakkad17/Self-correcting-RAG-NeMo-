"""Pydantic data models shared by scripts, the workflow, eval and (later) the API.

Corpus passages and eval items; the agents' plan, claims and verdicts; web fact checks;
guardrail events; the Controller's per-attempt trace and run state; the final SystemResult.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

ABSTAIN_TEXT = "insufficient evidence"

ItemType = Literal["answerable", "unanswerable", "adversarial_question", "adversarial_corpus"]
VerdictLabel = Literal["supported", "unsupported", "contradicted"]


class Passage(BaseModel):
    doc_id: str  # stable, citation-friendly id, e.g. "Scott_Derrickson"
    title: str
    text: str
    sentences: list[str]
    source: Literal["hotpotqa", "injected"] = "hotpotqa"


class EvalItem(BaseModel):
    qid: str
    split: Literal["dev", "test"]
    item_type: ItemType
    question: str
    answer: str | None  # None = unanswerable from the corpus; the system should abstain
    gold_doc_ids: list[str] = Field(default_factory=list)
    supporting_facts: list[tuple[str, int]] = Field(default_factory=list)  # (doc_id, sentence index)
    hotpot_type: str | None = None  # bridge / comparison
    canary: str | None = None  # adversarial items: the injection succeeded if this string is in the answer
    injected_doc_id: str | None = None  # adversarial_corpus: the poisoned passage


class RetrievedDoc(BaseModel):
    doc_id: str  # corpus id, or "web_<hash>" for a passage the agent found on the web
    title: str
    text: str
    score: float
    rank: int
    url: str = ""  # set for web passages


# ---- agents and Controller ----


class Plan(BaseModel):
    """Planner output: the question broken into retrievable sub-questions (1-4)."""
    sub_questions: list[str] = Field(min_length=1, max_length=4)


class Claim(BaseModel):
    """One factual statement in a drafted answer, with its citation."""
    claim_id: int
    text: str
    doc_id: str
    quote: str  # span copied from the cited passage


class Draft(BaseModel):
    """Researcher output: a short answer plus the claims that justify it."""
    answer: str
    claims: list[Claim] = Field(default_factory=list)


class Verdict(BaseModel):
    claim_id: int
    label: VerdictLabel
    reason: str = ""


class ResearchRequest(BaseModel):
    """Researcher input. `feedback` explains why the previous draft was rejected (empty on the first draft)."""
    question: str
    evidence: list[RetrievedDoc]
    sub_questions: list[str] = Field(default_factory=list)
    feedback: str = ""


class ResearchOutput(BaseModel):
    """Researcher result: the draft, plus passages the ReAct agent found with its tools (already sanitized)."""
    draft: Draft
    new_docs: list[RetrievedDoc] = Field(default_factory=list)
    searches: list[str] = Field(default_factory=list)  # e.g. "corpus: who founded X", "web: X founder"


class VerifyRequest(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    evidence: list[RetrievedDoc]


ANSWER_CLAIM_ID = 0  # the Verifier's extra check: "the claims together establish this answer to this question"


FactCheckStatus = Literal["confirmed", "contradicted", "not_found", "no_results", "error"]


class FactCheck(BaseModel):
    """Result of checking one claim against live web search results (Tavily), judged by an LLM."""
    claim_id: int
    query: str
    status: FactCheckStatus
    url: str = ""  # the web source the judge relied on
    source_title: str = ""
    detail: str = ""


class FactCheckRequest(BaseModel):
    claims: list[Claim]
    evidence: list[RetrievedDoc]


class FactCheckOutput(BaseModel):
    checks: list[FactCheck]


class VerifyOutput(BaseModel):
    verdicts: list[Verdict]
    fact_checks: list[FactCheck] = Field(default_factory=list)


class Action(str, Enum):
    ACCEPT = "accept"  # all claims supported -> output
    RETRIEVE = "retrieve"  # unsupported claims, retries left -> targeted re-retrieval, redraft
    REVISE = "revise"  # contradicted claims, retries left -> redraft from the evidence
    ABSTAIN = "abstain"  # retry cap reached without full support


class GuardrailEvent(BaseModel):
    """One guardrail decision, for the guardrail log in the UI and eval runs."""
    rail: Literal["input", "retrieval", "output", "abstention"]
    name: str  # the Colang flow
    action: Literal["passed", "blocked", "modified"]
    detail: str = ""


class Attempt(BaseModel):
    """One pass of the loop, for the trace shown in the UI and logged in eval runs."""
    index: int
    queries: list[str]  # queries run before this draft
    new_doc_ids: list[str]  # passages added to the evidence pool by those queries and by the agent's searches
    agent_searches: list[str] = Field(default_factory=list)  # tool calls the ReAct Researcher made for this draft
    draft: Draft
    verdicts: list[Verdict] = Field(default_factory=list)
    fact_checks: list[FactCheck] = Field(default_factory=list)
    action: Action


class RunState(BaseModel):
    """Everything the Controller knows about one question."""
    question: str
    retry_cap: int
    plan: Plan | None = None
    evidence: list[RetrievedDoc] = Field(default_factory=list)  # deduplicated pool, in arrival order
    attempts: list[Attempt] = Field(default_factory=list)
    retries_used: int = 0
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)  # retrieval-rail events during the loop

    def evidence_ids(self) -> set[str]:
        return {d.doc_id for d in self.evidence}


class ClaimWithVerdict(Claim):
    label: VerdictLabel | None = None
    reason: str = ""
    web_check: FactCheck | None = None  # live web (Tavily) check of the claim, when it ran


class SystemResult(BaseModel):
    """What the system returns for one question (also the core of the API response)."""
    system: str
    answer: str
    abstained: bool
    blocked: bool = False  # refused by an input rail before any retrieval
    retrieved: list[RetrievedDoc] = Field(default_factory=list)
    plan: Plan | None = None
    claims: list[ClaimWithVerdict] = Field(default_factory=list)
    answer_verdict: Verdict | None = None  # claim 0: do the claims establish the answer?
    attempts: list[Attempt] = Field(default_factory=list)
    retries: int = 0
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)


def system_result_to_str(result: SystemResult) -> str:
    """Toolkit type converter, so `nat run` / `nat serve` can print a SystemResult."""
    return result.model_dump_json(indent=2)


# ---- API (src/api) ----


class ApiSettings(BaseModel):
    """Per-request overrides. Omitted fields keep the values in configs/system.yml."""
    retry_cap: int | None = Field(default=None, ge=0, le=4)
    rails: dict[str, bool] = Field(default_factory=dict)  # e.g. {"pii": false}; keys: GuardrailSettings fields
    fact_check: bool | None = None  # live web (Tavily) check in the Verifier


class ApiRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    settings: ApiSettings = Field(default_factory=ApiSettings)


class ApiMetrics(BaseModel):
    latency_s: float
    llm_calls: int
    cached_llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    retries: int


class ApiResponse(BaseModel):
    """Everything the UI shows for one question."""
    question: str
    answer: str
    abstained: bool
    blocked: bool
    claims: list[ClaimWithVerdict]  # each with verdict, cited doc, quote and web check
    answer_verdict: Verdict | None
    plan: Plan | None
    attempts: list[Attempt]  # per-attempt trace: queries, new docs, draft, verdicts, Controller action
    retrieved: list[RetrievedDoc]  # the final evidence pool (after the retrieval rail)
    guardrail_events: list[GuardrailEvent]
    metrics: ApiMetrics
    settings: dict  # the effective settings this answer was produced with
