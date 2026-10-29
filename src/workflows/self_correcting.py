"""Systems C and D: Planner -> multi-query retrieval -> Researcher -> (Verifier -> Controller loop).

The Controller is plain Python over Pydantic state. Its policy (`decide`) is a pure function, and the loop
(`Controller.run`) takes the four agents as async callables, so both are unit-tested without any LLM.

The Verifier's verdicts include claim 0, "the claims establish this answer to this question", so a draft of
true but irrelevant claims is not accepted.

Policy, checked in this order after each verified draft:
  1. every claim supported (incl. 0)    -> ACCEPT  (answer goes to the output)
  2. retry cap reached                  -> ABSTAIN ("insufficient evidence")
  3. any claim contradicted             -> REVISE  (redraft from the evidence, contradicted claims named)
  4. otherwise (unsupported, or no claims at all) -> RETRIEVE (targeted queries from the unsupported claims)
Each REVISE or RETRIEVE uses one retry, so there are at most retry_cap + 1 drafts.
With no verifier (system C) the first draft is returned as is.
"""

from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import register_function

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
from src.state import Plan
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
ResearchFn = Callable[[ResearchRequest], Awaitable[Draft]]
VerifyFn = Callable[[VerifyRequest], Awaitable[list[Verdict]]]


class Controller:

    def __init__(self, research: ResearchFn, retrieve: RetrieveFn, plan: PlanFn | None = None,
                 verify: VerifyFn | None = None, retry_cap: int = 2, per_query_k: int = 5, max_evidence: int = 10,
                 per_claim_k: int = 3):
        self.research, self.retrieve, self.plan, self.verify = research, retrieve, plan, verify
        self.retry_cap, self.per_query_k, self.max_evidence, self.per_claim_k = (retry_cap, per_query_k,
                                                                                 max_evidence, per_claim_k)

    async def _retrieve_all(self, state: RunState, queries: list[str], k: int, limit: int | None) -> list[str]:
        rankings = [await self.retrieve(q, k, sorted(state.evidence_ids())) for q in queries]
        new_docs = merge_round_robin(rankings, state.evidence_ids(), limit)
        state.evidence.extend(new_docs)
        return [d.doc_id for d in new_docs]

    async def run(self, question: str) -> RunState:
        state = RunState(question=question, retry_cap=self.retry_cap)
        state.plan = await self.plan(question) if self.plan else Plan(sub_questions=[question])
        queries = list(dict.fromkeys([question] + state.plan.sub_questions))
        new_ids = await self._retrieve_all(state, queries, self.per_query_k, self.max_evidence)
        feedback = ""

        while True:
            draft = await self.research(ResearchRequest(question=question, evidence=state.evidence,
                                                        sub_questions=state.plan.sub_questions, feedback=feedback))
            if self.verify is None:  # system C: no verification, no loop
                state.attempts.append(Attempt(index=len(state.attempts), queries=queries, new_doc_ids=new_ids,
                                              draft=draft, action=Action.ACCEPT))
                return state

            verdicts = (await self.verify(VerifyRequest(question=question, answer=draft.answer, claims=draft.claims,
                                                        evidence=state.evidence))) if draft.claims else []
            action = decide(verdicts, state.retries_used, self.retry_cap)
            state.attempts.append(Attempt(index=len(state.attempts), queries=queries, new_doc_ids=new_ids,
                                          draft=draft, verdicts=verdicts, action=action))
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
    """Final output: the accepted draft, or an abstention. Claims carry their verdicts for the UI."""
    last = state.attempts[-1]
    labels = {v.claim_id: v for v in last.verdicts}
    claims = [ClaimWithVerdict(**c.model_dump(), label=labels[c.claim_id].label if c.claim_id in labels else None,
                               reason=labels[c.claim_id].reason if c.claim_id in labels else "")
              for c in last.draft.claims]
    if last.action == Action.ABSTAIN:
        answer, abstained = ABSTAIN_TEXT, True
    else:
        answer, abstained = last.draft.answer, ABSTAIN_TEXT in last.draft.answer.lower()
    return SystemResult(system=system_id, answer=answer, abstained=abstained, retrieved=state.evidence,
                        plan=state.plan, claims=claims, answer_verdict=labels.get(ANSWER_CLAIM_ID),
                        attempts=state.attempts, retries=state.retries_used)


class SelfCorrectingConfig(FunctionBaseConfig, name="self_correcting_workflow"):
    """Planner + multi-query retrieval + cited drafting, optionally verified in a bounded retry loop."""
    retriever: FunctionRef
    researcher: FunctionRef
    planner: FunctionRef | None = Field(default=None, description="Omit to retrieve with the question only.")
    verifier: FunctionRef | None = Field(default=None, description="Omit for system C (no verification loop).")
    retry_cap: int = Field(default=2, ge=0)
    per_query_k: int = Field(default=5, ge=1, description="Passages per query in the first retrieval.")
    max_evidence: int = Field(default=10, ge=1, description="Cap on the first evidence pool.")
    per_claim_k: int = Field(default=3, ge=1, description="New passages per targeted re-retrieval query.")
    system_id: str = "D"


@register_function(config_type=SelfCorrectingConfig)
async def self_correcting_workflow(config: SelfCorrectingConfig, builder: Builder):
    retriever = await builder.get_function(config.retriever)
    researcher = await builder.get_function(config.researcher)
    planner = await builder.get_function(config.planner) if config.planner else None
    verifier = await builder.get_function(config.verifier) if config.verifier else None

    async def retrieve(query: str, k: int, exclude: list[str]) -> list[RetrievedDoc]:
        out = await retriever.ainvoke(RetrievalRequest(query=query, top_k=k, exclude_ids=exclude),
                                      to_type=RetrievalOutput)
        return out.docs

    async def research(request: ResearchRequest) -> Draft:
        return await researcher.ainvoke(request, to_type=Draft)

    async def plan(question: str) -> Plan:
        return await planner.ainvoke(question, to_type=Plan)

    async def verify(request: VerifyRequest) -> list[Verdict]:
        return (await verifier.ainvoke(request, to_type=VerifyOutput)).verdicts

    controller = Controller(research, retrieve, plan if planner else None, verify if verifier else None,
                            retry_cap=config.retry_cap, per_query_k=config.per_query_k,
                            max_evidence=config.max_evidence, per_claim_k=config.per_claim_k)

    async def _answer(question: str) -> SystemResult:
        """Answer with evidence-backed claims, retrying, revising or abstaining as the Controller decides."""
        return to_result(await controller.run(question), config.system_id)

    yield FunctionInfo.from_fn(_answer, description=_answer.__doc__, converters=[system_result_to_str])
