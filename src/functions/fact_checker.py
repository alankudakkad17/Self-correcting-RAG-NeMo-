"""Web fact checker: an LLM judges each claim against live web search results (Tavily).

For every claim the Verifier marked supported:
1. search the web with the claim's text (cached);
2. pass the snippets through the retrieval rail: web text is untrusted and is about to reach a model;
3. one call to the judge model (the Verifier's model, not the Researcher's) labels every claim against its own
   snippets: supported / contradicted / not mentioned, citing the snippet it relied on.
Statuses: confirmed, contradicted, not_found (snippets do not address it), no_results (search found nothing),
error (search or judge failed; the Verifier then keeps the model's verdict).
"""

import asyncio

from pydantic import Field

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import FunctionRef
from nat.plugin_api import LLMRef
from nat.plugin_api import register_function

from src.functions.search_tools import sanitize
from src.llm import ChatClient
from src.parsing import parse_web_checks
from src.prompts import WEB_CHECK_SYSTEM
from src.prompts import web_check_user_prompt
from src.state import Claim
from src.state import FactCheck
from src.state import FactCheckOutput
from src.state import FactCheckRequest
from src.state import RetrievedDoc
from src.websearch import TavilyClient
from src.websearch import web_doc

STATUS = {"supported": "confirmed", "contradicted": "contradicted", "not_mentioned": "not_found"}


def to_fact_checks(searched: list[tuple[Claim, list[RetrievedDoc] | str]], judged: dict | None) -> list[FactCheck]:
    """Combine search results and the judge's labels into one FactCheck per claim (pure; unit-tested).

    `searched` holds (claim, docs) or (claim, error message); `judged` is parse_web_checks output, or None if the
    judge call itself failed.
    """
    checks = []
    for claim, docs in searched:
        base = {"claim_id": claim.claim_id, "query": claim.text}
        if isinstance(docs, str):
            checks.append(FactCheck(**base, status="error", detail=docs))
        elif not docs:
            checks.append(FactCheck(**base, status="no_results", detail="the web search found nothing"))
        elif judged is None or claim.claim_id not in judged:
            checks.append(FactCheck(**base, status="error", detail="the judge gave no valid verdict"))
        else:
            label, source, reason = judged[claim.claim_id]
            cited = docs[source - 1] if 1 <= source <= len(docs) else None
            checks.append(FactCheck(**base, status=STATUS[label], detail=reason,
                                    url=cited.url if cited else "", source_title=cited.title if cited else ""))
    return checks


class FactCheckerConfig(FunctionBaseConfig, name="tavily_fact_checker"):
    """Check claims against live web search results, judged by an LLM."""
    llm_name: LLMRef = Field(description="Judge model; use the Verifier's model, not the Researcher's.")
    guardrails: FunctionRef | None = Field(default=None, description="Retrieval rail for the web snippets.")
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", description="basic (1 credit) or advanced (2 credits)")
    cache: bool = Field(default=True, description="false = fresh search results every time (spends credits)")
    timeout_s: float = Field(default=30.0, gt=0)


@register_function(config_type=FactCheckerConfig)
async def tavily_fact_checker(config: FactCheckerConfig, builder: Builder):
    search = TavilyClient(search_depth=config.search_depth, max_results=config.max_results, timeout=config.timeout_s,
                          cache=config.cache)
    judge = ChatClient.from_toolkit_config(builder.get_llm_config(config.llm_name))
    rails = await builder.get_function(config.guardrails) if config.guardrails else None

    async def search_claim(claim: Claim) -> list[RetrievedDoc] | str:
        try:
            results = await search.search(claim.text)
        except Exception as e:  # network, key or credit failure: report it, do not guess
            return f"{type(e).__name__}: {e}"[:200]
        return await sanitize(rails, [web_doc(r, i) for i, r in enumerate(results, start=1)])

    async def _check(request: FactCheckRequest) -> FactCheckOutput:
        """Check every claim against live web search results."""
        searched = list(zip(request.claims, await asyncio.gather(*[search_claim(c) for c in request.claims])))
        to_judge = [(c, docs) for c, docs in searched if isinstance(docs, list) and docs]
        judged = {}
        if to_judge:
            try:
                reply = await judge.chat([{"role": "system", "content": WEB_CHECK_SYSTEM},
                                          {"role": "user", "content": web_check_user_prompt(to_judge)}])
                judged = parse_web_checks(reply.text, [c.claim_id for c, _ in to_judge])
            except Exception:  # judge unavailable: every judged claim becomes "error" (fail open)
                judged = None
        return FactCheckOutput(checks=to_fact_checks(searched, judged))

    yield FunctionInfo.from_fn(_check, description=_check.__doc__)
