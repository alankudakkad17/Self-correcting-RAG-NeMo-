"""Verifier: label each claim supported / unsupported / contradicted, then fact-check claims on the live web.

Uses a different model than the Researcher (see configs/system.yml). One LLM call per draft:
- every drafted claim against the passage it cites, and
- claim 0, "the claims together establish this answer to this question". Without it, a draft made of true
  but irrelevant claims (a similarly named entity, a missing hop) passes claim-by-claim checking.
Deterministic steps around the LLM:
- before: a claim citing a passage that is not in the evidence pool is "unsupported" without asking the model;
- after:  a "supported" claim that live web search results (Tavily) do not back is downgraded to
          "unsupported" (see apply_fact_checks). Search errors leave the model's verdict unchanged.
"""

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient
from src.parsing import parse_verdicts
from src.prompts import VERIFIER_SYSTEM
from src.prompts import answer_claim_text
from src.prompts import verifier_user_prompt
from src.state import ANSWER_CLAIM_ID
from src.state import Claim
from src.state import FactCheck
from src.state import FactCheckOutput
from src.state import FactCheckRequest
from src.state import Verdict
from src.state import VerifyOutput
from src.state import VerifyRequest


def apply_fact_checks(verdicts: list[Verdict], checks: list[FactCheck]) -> list[Verdict]:
    """Downgrade "supported" claims that the web search does not back. Other labels are left as they are."""
    by_claim = {c.claim_id: c for c in checks}
    out = []
    for v in verdicts:
        check = by_claim.get(v.claim_id)
        if v.label == "supported" and check and check.status in ("not_found", "no_results"):
            v = Verdict(claim_id=v.claim_id, label="unsupported", reason=f"Web check failed: {check.detail}")
        out.append(v)
    return out


class VerifierConfig(FunctionBaseConfig, name="verifier"):
    """Check every claim against the passages it cites, the answer against the claims, and claims on the web."""
    llm_name: LLMRef
    fact_checker: FunctionRef | None = None  # a `tavily_fact_checker` entry; omit to skip the web check


@register_function(config_type=VerifierConfig)
async def verifier(config: VerifierConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))
    fact_checker = await builder.get_function(config.fact_checker) if config.fact_checker else None

    async def _verify(request: VerifyRequest) -> VerifyOutput:
        """Label the answer (claim 0) and each claim supported, unsupported or contradicted."""
        known = {d.doc_id for d in request.evidence}
        missing = {c.claim_id for c in request.claims if c.doc_id not in known}
        to_check = [c for c in request.claims if c.claim_id not in missing]

        prompt = verifier_user_prompt(request.question, request.answer, to_check, request.evidence)
        result = await client.chat([{"role": "system", "content": VERIFIER_SYSTEM},
                                    {"role": "user", "content": prompt}])
        answer_claim = Claim(claim_id=ANSWER_CLAIM_ID, text=answer_claim_text(request.question, request.answer),
                             doc_id="", quote="")
        llm_verdicts = {v.claim_id: v for v in parse_verdicts(result.text, [answer_claim] + to_check)}

        verdicts = [llm_verdicts[ANSWER_CLAIM_ID]] + [
            Verdict(claim_id=c.claim_id, label="unsupported", reason=f"cited passage {c.doc_id!r} not in evidence")
            if c.claim_id in missing else llm_verdicts[c.claim_id] for c in request.claims
        ]

        checks = []
        supported = [c for c in to_check if llm_verdicts[c.claim_id].label == "supported"]
        if fact_checker and supported:
            out = await fact_checker.ainvoke(FactCheckRequest(claims=supported, evidence=request.evidence),
                                             to_type=FactCheckOutput)
            checks = out.checks
            verdicts = apply_fact_checks(verdicts, checks)
        return VerifyOutput(verdicts=verdicts, fact_checks=checks)

    yield FunctionInfo.from_fn(_verify, description=_verify.__doc__)
