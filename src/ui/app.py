"""Streamlit demo UI, built from Streamlit components only. Talks only to the API (src/ui/client.py); the NVIDIA
key never reaches this process. All retrieved and model text is escaped before it is shown.

Run:  streamlit run src/ui/app.py      (the API address comes from API_URL, default http://localhost:8000)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.ui.client import ApiClient  # noqa: E402
from src.ui.client import ApiError  # noqa: E402
from src.ui.client import RateLimited  # noqa: E402
from src.ui.components.render import ACTION_ICONS  # noqa: E402
from src.ui.components.render import ACTION_LABELS  # noqa: E402
from src.ui.components.render import RAIL_ACTION_COLORS  # noqa: E402
from src.ui.components.render import answer_with_citations  # noqa: E402
from src.ui.components.render import escape_md  # noqa: E402
from src.ui.components.render import highlight_quote  # noqa: E402
from src.ui.components.render import outcome  # noqa: E402
from src.ui.components.render import verdict_badge  # noqa: E402
from src.ui.components.render import web_badge  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
RAIL_LABELS = {"injection": "Block instructions hidden in questions", "off_topic": "Block off-topic requests",
               "unsafe": "Block harmful requests", "retrieval": "Strip instructions from sources",
               "groundedness": "Answer must follow from its claims", "citations": "Check citation format",
               "pii": "Hide personal data in answers", "abstention": "Explain when no answer is given"}
EXAMPLES = ["Which guitar brand was endorsed by the band centered by Ann and Nancy Wilson?",
            "What series did Tom Mandrake collaborate on that was based on a fictional superhero?",
            "Which country won the 2022 FIFA World Cup?"]

st.set_page_config(page_title="Self-Correcting RAG", page_icon=":material/fact_check:", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL)


client = get_client()


def use_example(question: str) -> None:
    st.session_state["question"] = question
    st.session_state["ask_now"] = True


# ---- sidebar: how answers are produced, and the user's documents ----

with st.sidebar:
    st.subheader(":material/tune: Answer settings")
    retry_cap = st.slider("Extra attempts before giving up", 0, 4, 2,
                          help="When a claim cannot be verified, search again or rewrite the draft up to this many times.")
    web_search = st.toggle("Search the web when needed", value=True,
                           help="Off: answers come only from the document collection and your uploads.")
    uploads = st.toggle("Search my documents", value=True)
    fact_check = st.toggle("Double-check claims on the web", value=True,
                           help="Claims from your uploaded documents are never sent to the web.")
    with st.expander("Safety checks", icon=":material/shield:"):
        rails = {key: st.toggle(label, value=True, key=f"rail_{key}") for key, label in RAIL_LABELS.items()}

    st.divider()
    st.subheader(":material/folder_open: My documents")
    picked = st.file_uploader("Add .txt, .md or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True,
                              key=f"uploader_{st.session_state.get('uploads_done', 0)}")
    if picked and st.button("Add to my documents", icon=":material/upload:", width="stretch"):
        with st.spinner("Reading and indexing..."):
            try:
                for r in client.upload_documents([(f.name, f.getvalue()) for f in picked]):
                    name = escape_md(r.get("filename") or "file")
                    if r["status"] == "rejected":
                        st.error(f"{name}: {escape_md(r.get('reason', 'rejected'))}")
                    else:
                        st.success(f"{name}: {r['status']}")
                st.session_state["uploads_done"] = st.session_state.get("uploads_done", 0) + 1  # clears the picker
            except ApiError as e:
                st.error(str(e))
    try:
        documents = client.list_documents()
    except ApiError:
        documents = []
    if not documents:
        st.caption("No documents yet. Added files become searchable right away.")
    for d in documents:
        name_col, del_col = st.columns([5, 1], vertical_alignment="center")
        name_col.markdown(f":material/description: {escape_md(d['filename'])}  \n"
                          f":gray[{d['chunks']} {'part' if d['chunks'] == 1 else 'parts'}]")
        if del_col.button("", icon=":material/delete:", key=f"del_{d['doc_id']}", help=f"Remove {d['filename']}"):
            try:
                client.delete_document(d["doc_id"])
                st.rerun()
            except ApiError as e:
                st.error(str(e))

    st.divider()
    if client.health():
        st.caption(":green[:material/circle:] Connected")
    else:
        st.error(f"Can't reach the answer service at {API_URL}. Start it with `uvicorn src.api.app:app`.")

# ---- question ----

st.title(":material/fact_check: Self-Correcting RAG", anchor=False)
st.markdown(":gray[Ask a factual question. You get an answer only when every claim checks out against its sources.]")

with st.form("ask", border=False):
    question = st.text_area("Question", key="question", height=90, label_visibility="collapsed",
                            placeholder="Ask a question about the collection, the web, or your documents")
    submitted = st.form_submit_button("Ask", type="primary", icon=":material/search:")

asking = (submitted or st.session_state.pop("ask_now", False)) and st.session_state.get("question", "").strip()
if not asking and not st.session_state.get("response"):
    st.markdown(":gray[Or try one of these:]")
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"example_{i}", on_click=use_example, args=(example,), icon=":material/arrow_outward:")

if asking:
    settings = {"retry_cap": retry_cap, "fact_check": fact_check, "web_search": web_search, "uploads": uploads,
                "rails": rails}
    with st.spinner("Planning, searching, drafting and checking each claim..."):
        try:
            st.session_state["response"] = client.ask(st.session_state["question"], settings)
        except RateLimited as e:
            st.warning(str(e))
        except ApiError as e:
            st.error(str(e))

response = st.session_state.get("response")
if not response:
    st.stop()

# ---- result: the stamp and the answer ----

result = outcome(response)
m = response["metrics"]
with st.container(border=True):
    st.badge(result["label"], icon=result["icon"], color=result["color"])
    if response["blocked"] or response["abstained"]:
        st.subheader(escape_md(result["detail"]), anchor=False)
        if response["abstained"]:
            st.caption("Try rephrasing, allowing more attempts, or turning on web search or your documents.")
    else:
        st.header(answer_with_citations(response["answer"], response["claims"]), anchor=False)
        st.caption(escape_md(result["detail"]))
    # Quiet run summary: the answer is the focus, not the numbers.
    attempts = f"{m['retries']} extra {'attempt' if m['retries'] == 1 else 'attempts'}"
    st.caption(f"Took {m['latency_s']:.1f} s, {m['llm_calls']} model calls ({m['cached_llm_calls']} from cache), "
               f"{attempts} and {m['prompt_tokens'] + m['completion_tokens']:,} tokens.")

tab_claims, tab_evidence, tab_trace, tab_rails, tab_sources = st.tabs(
    [":material/format_quote: Claims", ":material/plagiarism: Evidence", ":material/route: How it got there",
     ":material/shield: Safety log", ":material/library_books: Sources"])
docs = {d["doc_id"]: d for d in response["retrieved"]}

with tab_claims:
    if not response["claims"]:
        st.caption("No claims were drafted.")
    for i, c in enumerate(response["claims"], start=1):
        with st.container(border=True):
            st.markdown(f"**[{i}]** {escape_md(c['text'])}")
            source = docs.get(c["doc_id"], {})
            st.markdown(f"{verdict_badge(c.get('label'))} {web_badge(c.get('web_check'))} "
                        f":gray[from {escape_md(source.get('title') or c['doc_id'])}]")
            if c.get("quote"):
                st.caption(f"“{escape_md(c['quote'])}”")
            check = c.get("web_check") or {}
            if check.get("url"):
                st.caption(f"Web source: {escape_md(check.get('source_title') or '')} {escape_md(check['url'])}")
    if response.get("answer_verdict"):
        verdict = response["answer_verdict"]
        st.markdown(f"Does the answer follow from the claims? {verdict_badge(verdict['label'])}  \n"
                    f":gray[{escape_md(verdict['reason'])}]")

with tab_evidence:
    if not response["claims"]:
        st.caption("No claims to show evidence for.")
    else:
        idx = st.selectbox("Claim", range(len(response["claims"])),
                           format_func=lambda i: f"[{i + 1}] {response['claims'][i]['text'][:90]}")
        c = response["claims"][idx]
        doc = docs.get(c["doc_id"])
        if doc:
            with st.container(border=True):
                st.markdown(f"**{escape_md(doc['title'])}**")
                st.markdown(highlight_quote(doc["text"], c["quote"]))
                if doc.get("url"):
                    st.caption(escape_md(doc["url"]))
            st.caption("The highlighted span is the quote the claim relies on.")
        else:
            st.caption("The cited passage is not in the evidence.")

with tab_trace:
    if response["plan"]:
        st.markdown("**Broke the question into:**")
        st.markdown("\n".join(f"- {escape_md(s)}" for s in response["plan"]["sub_questions"]))
    for a in response["attempts"]:
        last = a is response["attempts"][-1]
        with st.expander(f"Attempt {a['index'] + 1}: {ACTION_LABELS.get(a['action'], a['action'])}",
                         icon=ACTION_ICONS.get(a["action"]), expanded=last):
            if a.get("agent_searches"):
                st.markdown("**The agent searched:**\n" + "\n".join(f"- {escape_md(s)}" for s in a["agent_searches"]))
            st.markdown(f"**Draft answer:** {escape_md(a['draft']['answer'])}")
            claim_text = {c["claim_id"]: c["text"] for c in a["draft"]["claims"]}
            for v in a["verdicts"]:
                what = "The answer follows from the claims" if v["claim_id"] == 0 else claim_text.get(v["claim_id"], "")
                st.markdown(f"{verdict_badge(v['label'])} {escape_md(what)}  \n:gray[{escape_md(v['reason'])}]")
            st.caption(f"{len(a['new_doc_ids'])} new sources in this attempt.")

with tab_rails:
    if not response["guardrail_events"]:
        st.caption("Safety checks were turned off for this answer.")
    for e in response["guardrail_events"]:
        color = RAIL_ACTION_COLORS.get(e["action"], "gray")
        st.markdown(f":{color}-badge[{e['action']}] **{escape_md(e['rail'])}**: {escape_md(e['name'])}"
                    + (f"  \n:gray[{escape_md(e['detail'])}]" if e.get("detail") else ""))

with tab_sources:
    st.caption(f"{len(response['retrieved'])} sources were considered.")
    for d in response["retrieved"]:
        icon = (":material/public:" if d.get("url") else
                ":material/description:" if d["doc_id"].startswith("up_") else ":material/menu_book:")
        with st.expander(escape_md(d["title"]), icon=icon):
            st.markdown(escape_md(d["text"]))
            if d.get("url"):
                st.caption(escape_md(d["url"]))
