"""Controller rules and loop (src/workflows/self_correcting.py). All agents are scripted fakes: no LLM."""

import pytest

from src.state import ABSTAIN_TEXT
from src.state import Action
from src.state import Claim
from src.state import Draft
from src.state import Plan
from src.state import RetrievedDoc
from src.state import FactCheck
from src.state import GuardrailEvent
from src.state import Verdict
from src.state import VerifyOutput
from src.workflows.self_correcting import Controller
from src.workflows.self_correcting import claims_report
from src.workflows.self_correcting import decide
from src.workflows.self_correcting import merge_round_robin
from src.workflows.self_correcting import targeted_queries
from src.workflows.self_correcting import to_result


def v(claim_id, label):
    return Verdict(claim_id=claim_id, label=label, reason=f"{label} reason")


def claim(i, text=None, doc="d1"):
    return Claim(claim_id=i, text=text or f"claim {i}", doc_id=doc, quote="q")


def doc(doc_id, rank=1):
    return RetrievedDoc(doc_id=doc_id, title=doc_id, text=f"text of {doc_id}", score=1.0, rank=rank)


# ---- decide(): the policy table ----

@pytest.mark.parametrize("labels, retries_used, cap, expected", [
    (["supported", "supported"], 0, 2, Action.ACCEPT),
    (["supported"], 2, 2, Action.ACCEPT),  # full support wins even at the cap
    (["supported", "unsupported"], 0, 2, Action.RETRIEVE),
    (["unsupported"], 1, 2, Action.RETRIEVE),
    (["supported", "contradicted"], 0, 2, Action.REVISE),
    (["unsupported", "contradicted"], 0, 2, Action.REVISE),  # contradiction is handled before missing support
    (["contradicted"], 2, 2, Action.ABSTAIN),  # cap reached
    (["unsupported"], 2, 2, Action.ABSTAIN),
    ([], 0, 2, Action.RETRIEVE),  # no claims at all: look for more evidence
    ([], 2, 2, Action.ABSTAIN),
    (["unsupported"], 0, 0, Action.ABSTAIN),  # cap 0: verify once, never retry
    (["supported"], 0, 0, Action.ACCEPT),
])
def test_decide(labels, retries_used, cap, expected):
    verdicts = [v(i + 1, label) for i, label in enumerate(labels)]
    assert decide(verdicts, retries_used, cap) == expected


def test_targeted_queries_use_unsupported_claim_text_only():
    claims = [claim(1, "A was born in X."), claim(2, "X is in Y."), claim(3, "A was born in X.")]
    verdicts = [v(1, "unsupported"), v(2, "supported"), v(3, "unsupported")]
    assert targeted_queries(claims, verdicts, fallback=["sub?"]) == ["A was born in X."]  # deduplicated


def test_targeted_queries_fall_back_to_sub_questions_and_respect_limit():
    assert targeted_queries([], [], fallback=["s1", "s2", "s3", "s4"], limit=3) == ["s1", "s2", "s3"]


def test_merge_round_robin_interleaves_dedupes_and_excludes():
    rankings = [[doc("a"), doc("b"), doc("c")], [doc("b"), doc("d")]]
    merged = merge_round_robin(rankings, exclude={"c"}, limit=None)
    assert [d.doc_id for d in merged] == ["a", "b", "d"]
    assert [d.doc_id for d in merge_round_robin(rankings, set(), limit=2)] == ["a", "b"]


# ---- the loop ----

class FakeAgents:
    """Scripted agents. drafts / verdict_lists are consumed one per call."""

    def __init__(self, drafts, verdict_lists=(), plan=None):
        self.drafts = list(drafts)
        self.verdict_lists = list(verdict_lists)
        self.plan_value = plan or Plan(sub_questions=["sub 1", "sub 2"])
        self.research_calls, self.verify_calls, self.retrieve_calls = [], [], []

    async def plan(self, question):
        return self.plan_value

    async def retrieve(self, query, k, exclude):
        self.retrieve_calls.append((query, k, list(exclude)))
        return [doc(f"{query}#{i}", rank=i) for i in range(1, k + 1)]

    async def research(self, request):
        self.research_calls.append(request)
        return self.drafts.pop(0)

    async def verify(self, request):
        self.verify_calls.append(request)
        return VerifyOutput(verdicts=self.verdict_lists.pop(0))


def make(agents, cap=2, sanitize=None):
    return Controller(agents.plan, agents.retrieve, agents.research, agents.verify, sanitize=sanitize, retry_cap=cap,
                      per_query_k=2, max_evidence=4, per_claim_k=2)


async def test_accepts_fully_supported_first_draft():
    agents = FakeAgents([Draft(answer="Paris", claims=[claim(1)])], [[v(1, "supported")]])
    state = await make(agents).run("Q?")
    assert [a.action for a in state.attempts] == [Action.ACCEPT]
    assert state.retries_used == 0
    result = to_result(state, "D")
    assert (result.answer, result.abstained) == ("Paris", False)
    assert result.claims[0].label == "supported"


async def test_first_retrieval_uses_question_plus_sub_questions_and_caps_pool():
    agents = FakeAgents([Draft(answer="x", claims=[claim(1)])], [[v(1, "supported")]])
    state = await make(agents).run("Q?")
    assert [q for q, _, _ in agents.retrieve_calls] == ["Q?", "sub 1", "sub 2"]
    assert len(state.evidence) == 4  # max_evidence
    assert [d.doc_id for d in state.evidence][:3] == ["Q?#1", "sub 1#1", "sub 2#1"]  # round robin


async def test_unsupported_claim_triggers_targeted_retrieval_then_accept():
    agents = FakeAgents(
        [Draft(answer="1713", claims=[claim(1, "B was born in 1713.")]),
         Draft(answer="1736", claims=[claim(1, "B was born in 1736.")])],
        [[v(1, "unsupported")], [v(1, "supported")]])
    state = await make(agents).run("Q?")

    assert [a.action for a in state.attempts] == [Action.RETRIEVE, Action.ACCEPT]
    assert state.retries_used == 1
    query, k, exclude = agents.retrieve_calls[-1]
    assert (query, k) == ("B was born in 1713.", 2)  # targeted: the claim text, per_claim_k
    assert set(exclude) == {d.doc_id for d in state.evidence[:4]}  # asks for passages not already held
    assert state.attempts[1].queries == ["B was born in 1713."]
    assert len(state.attempts[1].new_doc_ids) == 2 and len(state.evidence) == 6
    assert "B was born in 1713." in agents.research_calls[1].feedback
    assert to_result(state, "D").answer == "1736"


async def test_contradicted_claim_triggers_revision_without_new_retrieval():
    agents = FakeAgents(
        [Draft(answer="Gibson", claims=[claim(1), claim(2, "C makes Gibson guitars.")]),
         Draft(answer="Kramer", claims=[claim(1)])],
        [[v(1, "supported"), v(2, "contradicted")], [v(1, "supported")]])
    state = await make(agents).run("Q?")

    assert [a.action for a in state.attempts] == [Action.REVISE, Action.ACCEPT]
    assert len(agents.retrieve_calls) == 3  # only the first retrieval
    feedback = agents.research_calls[1].feedback
    assert "C makes Gibson guitars." in feedback and "contradicted reason" in feedback
    assert state.attempts[1].queries == [] and state.retries_used == 1


async def test_abstains_when_retry_cap_reached():
    agents = FakeAgents([Draft(answer=f"guess {i}", claims=[claim(1)]) for i in range(3)],
                        [[v(1, "unsupported")]] * 3)
    state = await make(agents, cap=2).run("Q?")

    assert [a.action for a in state.attempts] == [Action.RETRIEVE, Action.RETRIEVE, Action.ABSTAIN]
    assert len(agents.research_calls) == 3  # retry_cap + 1 drafts
    result = to_result(state, "D")
    assert (result.answer, result.abstained, result.retries) == (ABSTAIN_TEXT, True, 2)
    assert result.claims[0].label == "unsupported"  # the rejected claims stay visible in the result


async def test_retry_cap_zero_verifies_once():
    agents = FakeAgents([Draft(answer="x", claims=[claim(1)])], [[v(1, "contradicted")]])
    state = await make(agents, cap=0).run("Q?")
    assert [a.action for a in state.attempts] == [Action.ABSTAIN]


@pytest.mark.parametrize("cap", [0, 1, 2, 4])
async def test_loop_is_bounded_by_cap(cap):
    agents = FakeAgents([Draft(answer="x", claims=[claim(1)])] * (cap + 1), [[v(1, "contradicted")]] * (cap + 1))
    state = await make(agents, cap=cap).run("Q?")
    assert len(state.attempts) == cap + 1 and state.attempts[-1].action == Action.ABSTAIN


async def test_draft_without_claims_skips_verifier_and_retrieves_with_sub_questions():
    agents = FakeAgents([Draft(answer=ABSTAIN_TEXT), Draft(answer="y", claims=[claim(1)])], [[v(1, "supported")]])
    state = await make(agents).run("Q?")

    assert len(agents.verify_calls) == 1  # only the second draft had claims
    assert state.attempts[0].action == Action.RETRIEVE
    assert state.attempts[1].queries == ["sub 1", "sub 2"]
    assert "no supported answer" in agents.research_calls[1].feedback


async def test_retrieval_rail_sanitizes_every_passage_before_drafting():
    async def sanitize(docs):
        cleaned = [d.model_copy(update={"text": "clean"}) for d in docs]
        return cleaned, [GuardrailEvent(rail="retrieval", name="sanitize passage", action="modified")]

    agents = FakeAgents([Draft(answer="x", claims=[claim(1)])] * 2, [[v(1, "unsupported")], [v(1, "supported")]])
    state = await make(agents, sanitize=sanitize).run("Q?")
    assert all(d.text == "clean" for d in state.evidence)  # first pool and re-retrieved passages
    assert all(d.text == "clean" for d in agents.research_calls[0].evidence)
    assert len(state.guardrail_events) == 2  # one per retrieval round


async def test_fact_checks_reach_the_result_and_report():
    class Checked(FakeAgents):
        async def verify(self, request):
            return VerifyOutput(verdicts=[v(1, "supported")],
                                fact_checks=[FactCheck(claim_id=1, query="q", status="confirmed", score=97)])

    state = await make(Checked([Draft(answer="x", claims=[claim(1)])])).run("Q?")
    result = to_result(state, "S")
    assert result.claims[0].web_check.status == "confirmed"
    report = claims_report(state)
    assert report["accepted"] and report["claims"][0]["label"] == "supported"
    assert set(report["evidence_ids"]) == {d.doc_id for d in state.evidence}


async def test_accepted_abstaining_draft_is_reported_as_abstention():
    agents = FakeAgents([Draft(answer=ABSTAIN_TEXT, claims=[claim(1)])], [[v(1, "supported")]])
    result = to_result(await make(agents).run("Q?"), "D")
    assert result.abstained


# ---- claim 0: "the claims establish this answer" ----

async def test_true_but_irrelevant_claims_are_not_accepted():
    """Each claim is supported, but claim 0 (answer follows) is not: retrieve again, then abstain at the cap."""
    agents = FakeAgents([Draft(answer="earthenware", claims=[claim(1, "The Little Brown Jug is earthenware.")])] * 2,
                        [[v(0, "unsupported"), v(1, "supported")]] * 2)
    state = await make(agents, cap=1).run("What is the Little Brown Stein made of?")

    assert [a.action for a in state.attempts] == [Action.RETRIEVE, Action.ABSTAIN]
    assert state.attempts[1].queries == ["sub 1", "sub 2"]  # no unsupported claim text: re-query the plan
    feedback = agents.research_calls[1].feedback
    assert "do not establish the answer" in feedback and "unsupported reason" in feedback
    assert agents.verify_calls[0].answer == "earthenware"
    result = to_result(state, "D")
    assert result.abstained and result.answer_verdict.label == "unsupported"
    assert [c.claim_id for c in result.claims] == [1]  # claim 0 is reported separately, not as a claim


async def test_contradicted_answer_triggers_revision():
    agents = FakeAgents([Draft(answer="Gibson", claims=[claim(1)]), Draft(answer="Kramer", claims=[claim(1)])],
                        [[v(0, "contradicted"), v(1, "supported")], [v(0, "supported"), v(1, "supported")]])
    state = await make(agents).run("Q?")
    assert [a.action for a in state.attempts] == [Action.REVISE, Action.ACCEPT]
    assert "contradicted reason" in agents.research_calls[1].feedback
    assert to_result(state, "D").answer_verdict.label == "supported"
