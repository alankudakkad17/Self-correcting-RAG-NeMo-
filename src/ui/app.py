"""Streamlit demo UI. Talks only to the API (src/ui/client.py); the NVIDIA key never reaches this process.

Run:  streamlit run src/ui/app.py      (API_URL defaults to http://localhost:8000)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui.client import ApiClient  # noqa: E402
from src.ui.client import ApiError  # noqa: E402
from src.ui.client import RateLimited  # noqa: E402
from src.ui.components.render import VERDICT_COLORS  # noqa: E402
from src.ui.components.render import WIKI_LABELS  # noqa: E402
from src.ui.components.render import answer_with_citations  # noqa: E402
from src.ui.components.render import claim_rows  # noqa: E402
from src.ui.components.render import escape_md  # noqa: E402
from src.ui.components.render import highlight_quote  # noqa: E402
from src.ui.components.render import verdict_badge  # noqa: E402

RAIL_LABELS = {"injection": "Prompt injection (input)", "off_topic": "Off-topic (input)",
               "unsafe": "Unsafe request (input)", "retrieval": "Sanitize passages (retrieval)",
               "groundedness": "Groundedness (output)", "citations": "Citation format (output)",
               "pii": "PII redaction (output)", "abstention": "Safe abstention message"}

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")


@st.cache_resource
def get_client(url: str) -> ApiClient:
    return ApiClient(url)


@st.cache_data(ttl=60)
def get_examples(url: str) -> list[dict]:
    try:
        return get_client(url).examples()
    except ApiError:
        return []


# ---- sidebar: connection and settings ----

with st.sidebar:
    api_url = st.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000"))
    client = get_client(api_url)
    st.caption("API online" if client.health() else "API not reachable: start it with `uvicorn src.api.app:app`")

    st.subheader("Settings")
    retry_cap = st.slider("Retry cap", 0, 4, 2, help="Re-retrieve / revise rounds before abstaining")
    fact_check = st.toggle("Wikipedia fact check", value=True)
    with st.expander("Guardrails"):
        rails = {key: st.toggle(label, value=True, key=f"rail_{key}") for key, label in RAIL_LABELS.items()}

    examples = get_examples(api_url)
    st.subheader("Cached examples")
    example_titles = ["(none)"] + [e["question"][:70] for e in examples]
    picked = st.selectbox("Load a stored answer (no live calls)", range(len(example_titles)),
                          format_func=lambda i: example_titles[i])

# ---- question ----

st.title("Self-Correcting RAG")
st.caption("Answers only when every claim is backed by evidence: plan, retrieve, draft with citations, verify "
           "(different model + live Wikipedia), then accept, retry, revise or abstain.")

default_question = examples[picked - 1]["question"] if picked else ""
question = st.text_area("Question", value=default_question, height=80)
col_ask, col_cached = st.columns([1, 1])
ask_clicked = col_ask.button("Ask", type="primary", disabled=not question.strip())
cached_clicked = col_cached.button("Show cached answer", disabled=not picked)

if ask_clicked:
    settings = {"retry_cap": retry_cap, "fact_check": fact_check, "rails": rails}
    with st.spinner("Planning, retrieving, drafting and verifying..."):
        try:
            st.session_state["response"] = client.ask(question, settings)
        except RateLimited as e:
            st.warning(f"Rate limit: {e}")
        except ApiError as e:
            st.error(str(e))
if cached_clicked:
    st.session_state["response"] = {**examples[picked - 1]["response"], "cached_example": True}

response = st.session_state.get("response")
if not response:
    st.stop()

# ---- answer ----

if response.get("cached_example"):
    st.info("Cached example: stored response, not computed now.")
if response["blocked"]:
    st.error(f"Blocked by an input guardrail. {response['answer']}")
elif response["abstained"]:
    st.warning(f"Abstained. {response['answer']}")
else:
    st.markdown(f"### {answer_with_citations(response['answer'], response['claims'])}")

m = response["metrics"]
final_action = response["attempts"][-1]["action"] if response["attempts"] else ("blocked" if response["blocked"] else "-")
cols = st.columns(5)
cols[0].metric("Latency", f"{m['latency_s']:.1f} s")
cols[1].metric("LLM calls", m["llm_calls"], help=f"{m['cached_llm_calls']} served from cache")
cols[2].metric("Tokens", m["prompt_tokens"] + m["completion_tokens"])
cols[3].metric("Retries", m["retries"])
cols[4].metric("Controller", final_action)

tab_claims, tab_evidence, tab_trace, tab_rails, tab_passages = st.tabs(
    ["Claims", "Evidence", "Loop trace", "Guardrail log", "Plan and passages"])

with tab_claims:
    if response["claims"]:
        table = pd.DataFrame(claim_rows(response["claims"]))
        st.dataframe(table.style.map(lambda label: f"color: {VERDICT_COLORS.get(label, 'gray')}", subset=["verdict"]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No claims were drafted.")
    if response.get("answer_verdict"):
        verdict = response["answer_verdict"]
        st.markdown(f"Answer follows from the claims: {verdict_badge(verdict['label'])} {escape_md(verdict['reason'])}")

with tab_evidence:
    docs = {d["doc_id"]: d for d in response["retrieved"]}
    if response["claims"]:
        idx = st.selectbox("Claim", range(len(response["claims"])),
                           format_func=lambda i: f"[{i + 1}] {response['claims'][i]['text'][:90]}")
        c = response["claims"][idx]
        st.markdown(f"{verdict_badge(c.get('label'))} cites **{escape_md(c['doc_id'])}** Â· "
                    f"{escape_md(WIKI_LABELS.get((c.get('wikipedia') or {}).get('status')))}")
        doc = docs.get(c["doc_id"])
        if doc:
            st.markdown(f"**{escape_md(doc['title'])}**")
            st.markdown(highlight_quote(doc["text"], c["quote"]))
        else:
            st.caption("The cited passage is not in the evidence pool.")
    else:
        st.caption("No claims to show evidence for.")

with tab_trace:
    if response["plan"]:
        st.markdown("**Plan:** " + escape_md(" | ".join(response["plan"]["sub_questions"])))
    for a in response["attempts"]:
        with st.expander(f"Attempt {a['index'] + 1}: {a['action']}", expanded=a is response["attempts"][-1]):
            if a["queries"]:
                st.markdown("**Queries:** " + escape_md(" | ".join(a["queries"])))
            st.markdown(f"**New passages:** {escape_md(', '.join(a['new_doc_ids']) or 'none')}")
            st.markdown(f"**Draft answer:** {escape_md(a['draft']['answer'])}")
            labels = {v["claim_id"]: v for v in a["verdicts"]}
            rows = [{"claim": "the answer follows (claim 0)" if cid == 0 else
                     next((c["text"] for c in a["draft"]["claims"] if c["claim_id"] == cid), ""),
                     "verdict": v["label"], "reason": v["reason"]} for cid, v in labels.items()]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_rails:
    if response["guardrail_events"]:
        st.dataframe(pd.DataFrame(response["guardrail_events"]), hide_index=True, width="stretch")
    else:
        st.caption("No guardrail events (guardrails off).")

with tab_passages:
    for d in response["retrieved"]:
        with st.expander(f"{d['rank']}. {escape_md(d['title'])}"):
            st.markdown(escape_md(d["text"]))
