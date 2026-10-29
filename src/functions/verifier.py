"""Verifier: label each claim supported / unsupported / contradicted. One LLM call per draft (all claims at once).

Uses a different model than the Researcher (see configs). Besides the drafted claims it always checks
claim 0, "the claims together establish this answer to this question". Without it, a draft made of true
but irrelevant claims (a similarly named entity, a missing hop) passes claim-by-claim checking.

Deterministic pre-check before the LLM: a claim citing a passage that is not in the evidence pool is
"unsupported" without asking the model. (Quote-level checking is the MCP citation checker's job, system G.)
"""

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.llm import ChatClient
from src.parsing import parse_verdicts
from src.prompts import VERIFIER_SYSTEM
from src.prompts import answer_claim_text
from src.prompts import verifier_user_prompt
from src.state import ANSWER_CLAIM_ID
from src.state import Claim
from src.state import Verdict
from src.state import VerifyOutput
from src.state import VerifyRequest


class VerifierConfig(FunctionBaseConfig, name="verifier"):
    """Check every claim against the passages it cites, and the answer against the claims."""
    llm_name: LLMRef


@register_function(config_type=VerifierConfig)
async def verifier(config: VerifierConfig, builder: Builder):
    client = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))

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
        return VerifyOutput(verdicts=verdicts)

    yield FunctionInfo.from_fn(_verify, description=_verify.__doc__)
