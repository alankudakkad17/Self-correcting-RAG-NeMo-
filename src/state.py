"""Pydantic data models shared by scripts, workflows, eval and (later) the API.

Milestone 2: corpus passages, eval items, retrieved docs.
Milestone 3: plan, claims, verdicts, per-attempt trace and the Controller's run state.
ApiResponse is added in milestone 6.
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
    doc_id: str
    title: str
    text: str
    score: float
    rank: int


# ---- Milestone 3: agents and Controller ----


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


class VerifyRequest(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    evidence: list[RetrievedDoc]


ANSWER_CLAIM_ID = 0  # the Verifier's extra check: "the claims together establish this answer to this question"


class VerifyOutput(BaseModel):
    verdicts: list[Verdict]


class Action(str, Enum):
    ACCEPT = "accept"  # all claims supported -> output
    RETRIEVE = "retrieve"  # unsupported claims, retries left -> targeted re-retrieval, redraft
    REVISE = "revise"  # contradicted claims, retries left -> redraft from the evidence
    ABSTAIN = "abstain"  # retry cap reached without full support


class Attempt(BaseModel):
    """One pass of the loop, for the trace shown in the UI and logged in eval runs."""
    index: int
    queries: list[str]  # queries run before this draft
    new_doc_ids: list[str]  # passages added to the evidence pool by those queries
    draft: Draft
    verdicts: list[Verdict] = Field(default_factory=list)
    action: Action


class RunState(BaseModel):
    """Everything the Controller knows about one question."""
    question: str
    retry_cap: int
    plan: Plan | None = None
    evidence: list[RetrievedDoc] = Field(default_factory=list)  # deduplicated pool, in arrival order
    attempts: list[Attempt] = Field(default_factory=list)
    retries_used: int = 0

    def evidence_ids(self) -> set[str]:
        return {d.doc_id for d in self.evidence}


class ClaimWithVerdict(Claim):
    label: VerdictLabel | None = None  # None when the system does not verify (C)
    reason: str = ""


class SystemResult(BaseModel):
    """What every system (H, A, B, C, D, ...) returns for one question."""
    system: str
    answer: str
    abstained: bool
    retrieved: list[RetrievedDoc] = Field(default_factory=list)
    # Filled by the self-correcting systems (C, D, ...)
    plan: Plan | None = None
    claims: list[ClaimWithVerdict] = Field(default_factory=list)
    answer_verdict: Verdict | None = None  # claim 0: do the claims establish the answer?
    attempts: list[Attempt] = Field(default_factory=list)
    retries: int = 0


def system_result_to_str(result: SystemResult) -> str:
    """Toolkit type converter, so `nat run` / `nat serve` can print a SystemResult."""
    return result.model_dump_json(indent=2)
