"""The system: input rails -> Planner -> retrieval (+ retrieval rail) -> Researcher -> Verifier (+ web check)
-> Controller loop -> output / abstention rails.

The Controller is plain Python over Pydantic state. Its policy (`decide`) is a pure function, and the loop
(`Controller.run`) takes the agents as async callables, so both are unit-tested without any LLM.

The Verifier's verdicts include claim 0, "the claims establish this answer to this question", so a draft of
true but irrelevant claims is not accepted.

Policy, checked in this order after each verified draft:
  1. every claim supported (incl. 0)    -> ACCEPT  (answer goes to the output rails)
  2. retry cap reached                  -> ABSTAIN (abstention rail: "insufficient evidence")
  3. any claim contradicted             -> REVISE  (redraft from the evidence, contradicted claims named)
  4. otherwise (unsupported, or no claims at all) -> RETRIEVE (targeted queries from the unsupported claims)
Each REVISE or RETRIEVE uses one retry, so there are at most retry_cap + 1 drafts.
"""

from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import register_function

from src.functions.guardrails import GuardrailRequest
from src.functions.guardrails import GuardrailResponse
from src.functions.retriever import RetrievalOutput
from src.functions.retriever import RetrievalRequest
from src.prompts import answer_feedback
from src.prompts import contradicted_feedback
from src.prompts import unsupported_feedback
from src.state import ABSTAIN_TEXT
from src.state import ANSWER_CLAIM_ID
from src.state import Action
from src.state import Attempt
from src.state import Claim
from src.state import ClaimWithVerdict
from src.state import Draft
from src.state import GuardrailEvent
from src.state import Plan
from src.state import ResearchOutput
from src.state import ResearchRequest
from src.state import RetrievedDoc
from src.state import RunState
from src.state import SystemResult
from src.state import Verdict
from src.state import VerifyOutput
from src.state import VerifyRequest
from src.state import system_result_to_str

NO_CLAIMS_FEEDBACK = "Your previous draft gave no supported answer. More passages were retrieved; try again."


def decide(verdicts: list[Verdict], retries_used: int, retry_cap: int) -> Action:
    """The Controller policy. `verdicts` is empty when the draft made no claims."""
    if verdicts and all(v.label == "supported" for v in verdicts):
        return Action.ACCEPT
    if retries_used >= retry_cap:
        return Action.ABSTAIN
    if any(v.label == "contradicted" for v in verdicts):
        return Action.REVISE
    return Action.RETRIEVE


def targeted_queries(claims: list[Claim], verdicts: list[Verdict], fallback: list[str], limit: int = 3) -> list[str]:
    """Queries for re-retrieval: the text of each unsupported claim. Falls back to the plan's sub-questions."""
    unsupported = {v.claim_id for v in verdicts if v.label == "unsupported"}
    queries = [c.text for c in claims if c.claim_id in unsupported] or list(fallback)
    return list(dict.fromkeys(q for q in queries if q.strip()))[:limit]  # dedupe, keep order


def merge_round_robin(rankings: list[list[RetrievedDoc]], exclude: set[str], limit: int | None) -> list[RetrievedDoc]:
    """Interleave per-query rankings (rank 1 of each query, then rank 2, ...), dropping duplicates."""
    merged, seen = [], set(exclude)
    for rank in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking) and ranking[rank].doc_id not in seen:
                seen.add(ranking[rank].doc_id)
                merged.append(ranking[rank])
    return merged[:limit] if limit else merged


PlanFn = Callable[[str], Awaitable[Plan]]
RetrieveFn = Callable[[str, int, list[str]], Awaitable[list[RetrievedDoc]]]
ResearchFn = Callable[[ResearchRequest], Awaitable[ResearchOutput]]
VerifyFn = Callable[[VerifyRequest], Awaitable[VerifyOutput]]
SanitizeFn = Callable[[list[RetrievedDoc]], Awaitable[tuple[list[RetrievedDoc], list[GuardrailEvent]]]]


class Controller:

    def __init__(self, plan: PlanFn, retrieve: RetrieveFn, research: ResearchFn, verify: VerifyFn,
                 sanitize: SanitizeFn | None = None, retry_cap: int = 2, per_query_k: int = 5, max_evidence: int = 10,
                 per_claim_k: int = 3):
        self.plan, self.retrieve, self.research, self.verify, self.sanitize = plan, retrieve, research, verify, sanitize
        self.retry_cap, self.per_query_k, self.max_evidence, self.per_claim_k = (retry_cap, per_query_k,
                                                                                 max_evidence, per_claim_k)

    async def _retrieve_all(self, state: RunState, queries: list[str], k: int, limit: int | None) -> list[str]:
        rankings = [await self.retrieve(q, k, sorted(state.evidence_ids())) for q in queries]
        new_docs = merge_round_robin(rankings, state.evidence_ids(), limit)
        if self.sanitize:  # retrieval rail: passages are untrusted until sanitized
            new_docs, events = await self.sanitize(new_docs)
            state.guardrail_events.extend(events)
        state.evidence.extend(new_docs)
        return [d.doc_id for d in new_docs]

    async def run(self, question: str) -> RunState:
        state = RunState(question=question, retry_cap=self.retry_cap)
        state.plan = await self.plan(question)
        queries = list(dict.fromkeys([question] + state.plan.sub_questions))
        new_ids = await self._retrieve_all(state, queries, self.per_query_k, self.max_evidence)
        feedback = ""

        while True:
            researched = await self.research(ResearchRequest(question=question, evidence=state.evidence,
                                                             sub_questions=state.plan.sub_questions, feedback=feedback))
            draft = researched.draft
            # Passages the ReAct Researcher found with its tools join the pool (already sanitized by the tools),
            # so the Verifier can check claims that cite them.
            held = state.evidence_ids()
            found = [d for d in researched.new_docs if d.doc_id not in held]
            state.evidence.extend(found)
            new_ids = new_ids + [d.doc_id for d in found]
            checked = (await self.verify(VerifyRequest(question=question, answer=draft.answer, claims=draft.claims,
                                                       evidence=state.evidence))) if draft.claims else VerifyOutput(
                                                           verdicts=[])
            verdicts = checked.verdicts
            action = decide(verdicts, state.retries_used, self.retry_cap)
            state.attempts.append(Attempt(index=len(state.attempts), queries=queries, new_doc_ids=new_ids,
                                          agent_searches=researched.searches, draft=draft, verdicts=verdicts,
                                          fact_checks=checked.fact_checks, action=action))
            if action in (Action.ACCEPT, Action.ABSTAIN):
                return state

            state.retries_used += 1
            if action == Action.REVISE:
                queries, new_ids = [], []
            else:  # RETRIEVE: unsupported claim texts, else the sub-questions again (new passages only)
                queries = targeted_queries(draft.claims, verdicts, state.plan.sub_questions)
                new_ids = await self._retrieve_all(state, queries, self.per_claim_k, None)
            by_id = {v.claim_id: v for v in verdicts}
            feedback = self._feedback(action, draft.claims, by_id, by_id.get(ANSWER_CLAIM_ID))

    @staticmethod
    def _feedback(action: Action, claims: list[Claim], by_id: dict[int, Verdict], answer_check: Verdict | None) -> str:
        """Why the draft was rejected, for the Researcher's next attempt."""
        label = "contradicted" if action == Action.REVISE else "unsupported"
        bad = [c for c in claims if by_id[c.claim_id].label == label]
        parts = []
        if bad:
            parts.append(contradicted_feedback(bad, by_id) if action == Action.REVISE else unsupported_feedback(bad))
        if answer_check and answer_check.label != "supported":
            parts.append(answer_feedback(answer_check))
        return "\n\n".join(parts) or NO_CLAIMS_FEEDBACK


def to_result(state: RunState, system_id: str) -> SystemResult:
    """Final output before the output rails: the accepted draft, or an abstention. Claims carry their verdicts."""
    last = state.attempts[-1]
    labels = {v.claim_id: v for v in last.verdicts}
    checks = {c.claim_id: c for c in last.fact_checks}
    claims = [ClaimWithVerdict(**c.model_dump(), label=labels[c.claim_id].label if c.claim_id in labels else None,
                               reason=labels[c.claim_id].reason if c.claim_id in labels else "",
                               web_check=checks.get(c.claim_id))
              for c in last.draft.claims]
    if last.action == Action.ABSTAIN:
        answer, abstained = ABSTAIN_TEXT, True
    else:
        answer, abstained = last.draft.answer, ABSTAIN_TEXT in last.draft.answer.lower()
    return SystemResult(system=system_id, answer=answer, abstained=abstained, retrieved=state.evidence,
                        plan=state.plan, claims=claims, answer_verdict=labels.get(ANSWER_CLAIM_ID),
                        attempts=state.attempts, retries=state.retries_used,
                        guardrail_events=list(state.guardrail_events))


def claims_report(state: RunState) -> dict:
    """What the output rails check: was the answer accepted, and is every claim supported and well cited?"""
    last = state.attempts[-1]
    labels = {v.claim_id: v.label for v in last.verdicts}
    return {"accepted": last.action == Action.ACCEPT,
            "claims": [{"text": c.text, "doc_id": c.doc_id, "quote": c.quote, "label": labels.get(c.claim_id)}
                       for c in last.draft.claims],
            "answer_label": labels.get(ANSWER_CLAIM_ID),
            "evidence_ids": sorted(state.evidence_ids())}


class SelfCorrectingConfig(FunctionBaseConfig, name="self_correcting_workflow"):
    """Evidence-backed QA: plan, retrieve, draft with citations, verify, retry / revise / abstain, guardrails."""
    retriever: FunctionRef
    planner: FunctionRef
    researcher: FunctionRef
    verifier: FunctionRef
    guardrails: FunctionRef | None = Field(default=None, description="Omit to run without NeMo Guardrails.")
    uploads: FunctionRef | None = Field(default=None, description="An `uploads_retriever`: relevant chunks of the "
                                        "user's uploaded documents join every retrieval. Omit for corpus only.")
    retry_cap: int = Field(default=2, ge=0)
    per_query_k: int = Field(default=5, ge=1, description="Passages per query in the first retrieval.")
    max_evidence: int = Field(default=10, ge=1, description="Cap on the first evidence pool.")
    per_claim_k: int = Field(default=3, ge=1, description="New passages per targeted re-retrieval query.")
    system_id: str = "self-correcting-rag"


@register_function(config_type=SelfCorrectingConfig)
async def self_correcting_workflow(config: SelfCorrectingConfig, builder: Builder):
    retriever = await builder.get_function(config.retriever)
    planner = await builder.get_function(config.planner)
    researcher = await builder.get_function(config.researcher)
    verifier = await builder.get_function(config.verifier)
    rails = await builder.get_function(config.guardrails) if config.guardrails else None
    uploads = await builder.get_function(config.uploads) if config.uploads else None

    async def guard(**kwargs) -> GuardrailResponse:
        return await rails.ainvoke(GuardrailRequest(**kwargs), to_type=GuardrailResponse)

    async def retrieve(query: str, k: int, exclude: list[str]) -> list[RetrievedDoc]:
        request = RetrievalRequest(query=query, top_k=k, exclude_ids=exclude)
        out = await retriever.ainvoke(request, to_type=RetrievalOutput)
        if uploads is None:
            return out.docs
        # Relevant uploaded chunks go first: only chunks above the similarity cut-off are returned at all.
        found = await uploads.ainvoke(request, to_type=RetrievalOutput)
        return found.docs + out.docs

    async def sanitize(docs: list[RetrievedDoc]) -> tuple[list[RetrievedDoc], list[GuardrailEvent]]:
        out = await guard(stage="passages", question="", docs=docs)
        return out.docs, out.events

    controller = Controller(
        plan=lambda q: planner.ainvoke(q, to_type=Plan),
        retrieve=retrieve,
        research=lambda request: researcher.ainvoke(request, to_type=ResearchOutput),
        verify=lambda request: verifier.ainvoke(request, to_type=VerifyOutput),
        sanitize=sanitize if rails else None,
        retry_cap=config.retry_cap, per_query_k=config.per_query_k, max_evidence=config.max_evidence,
        per_claim_k=config.per_claim_k)

    async def _answer(question: str) -> SystemResult:
        """Answer with evidence-backed claims, retrying, revising or abstaining as the Controller decides."""
        events: list[GuardrailEvent] = []
        if rails:
            checked = await guard(stage="input", question=question)
            events += checked.events
            if not checked.allowed:
                return SystemResult(system=config.system_id, answer=checked.text, abstained=True, blocked=True,
                                    guardrail_events=events)

        state = await controller.run(question)
        result = to_result(state, config.system_id)
        events += result.guardrail_events

        if rails and not result.abstained:
            checked = await guard(stage="output", question=question, answer=result.answer,
                                  report=claims_report(state))
            events += checked.events
            if checked.allowed:
                result.answer = checked.text
            else:
                result.abstained = True
        if rails and result.abstained:
            checked = await guard(stage="abstention", question=question, answer=ABSTAIN_TEXT)
            events += checked.events
            result.answer = checked.text
        result.guardrail_events = events
        return result

    yield FunctionInfo.from_fn(_answer, description=_answer.__doc__, converters=[system_result_to_str])
