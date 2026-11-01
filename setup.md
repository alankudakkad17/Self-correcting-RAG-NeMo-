# Setup (Windows, native Python)

Tested with Python 3.13.14 (the toolkit needs >=3.11,<3.14). On this machine `py -3.11` resolves to
Anaconda 3.10, so use `py -3.13`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps   # registers src/register.py as a toolkit plugin
copy .env.example .env                                      # then fill in NVIDIA_API_KEY and TAVILY_API_KEY
```

Behind Kaspersky HTTPS scanning, add to every pip command:
`--trusted-host pypi.org --trusted-host files.pythonhosted.org`

## The system

One workflow, `configs/system.yml`:

input rails -> Planner -> hybrid retrieval (BM25 + FAISS -> RRF -> hosted Nemotron rerank) + retrieval rail
-> Researcher (cited draft) -> Verifier (different model, plus claim 0 "the claims establish the answer")
+ live web fact check (Tavily) -> Controller: accept / retrieve again / revise / abstain (retry cap 2)
-> output rails (groundedness, citation format, PII) or abstention rail.

Every guardrail can be switched off in `configs/system.yml`.

## Build the data and index (once)

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py     # pinned HotpotQA distractor validation file (27 MB)
.\.venv\Scripts\python.exe scripts/make_eval_sets.py    # frozen corpus + dev/test/pilot; refuses to overwrite
.\.venv\Scripts\python.exe scripts/build_index.py       # embeds the corpus, builds data/index/ (~80 s)
```

## Run

```powershell
# One question (JSON: answer, claims with verdicts and web checks, per-attempt trace, guardrail log)
.\.venv\Scripts\nat.exe run --config_file configs/system.yml --input "<question>"

# Unit tests (no API calls)
.\.venv\Scripts\python.exe -m pytest -q

# Check every configured model answers
.\.venv\Scripts\python.exe scripts/smoke_models.py
```

## Demo: API + Streamlit UI

```powershell
# Windows, no Docker / make: API in a new window, UI here -> http://localhost:8501
.\demo.ps1

# or each by hand
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --port 8000
.\.venv\Scripts\python.exe -m streamlit run src/ui/app.py

# with Docker + make (after building data/index on the host)
make demo
```

The API (`src/api/app.py`) holds the NVIDIA key; the UI only knows `API_URL`. `POST /ask` takes
`{"question": ..., "settings": {"retry_cap": 0-4, "rails": {"pii": false, ...}, "fact_check": true}}` and returns
the answer, claims with verdicts / cited quotes / web checks and sources, the per-attempt trace, guardrail events and
latency / token / LLM-call counts. Rate limits come back as HTTP 429 and show as a warning in the UI.

## Evaluation (RAGAS only, to be finalised)

```powershell
.\.venv\Scripts\python.exe eval/run_eval.py --split pilot             # records outputs to eval/results/runs/
.\.venv\Scripts\python.exe eval/ragas_eval.py --run eval/results/runs/<run_dir>
```

## The ReAct Researcher

The Researcher is the toolkit's built-in ReAct agent (`research_agent` in `configs/system.yml`, `_type: react_agent`)
with two tools, `corpus_search` and `web_search`. It comes from the toolkit's LangChain plugin
(`nvidia-nat-langchain[nvidia]`, in requirements.txt). To use the plain single-call Researcher instead, set
`researcher: {_type: researcher, llm_name: generator}`.

## Known version workarounds
- `nvidia-nat[guardrails]` is not used: it pins `nemoguardrails<0.22`, which needs `annoy`
  (no Windows wheel). We install `nemoguardrails==0.24.1` on its own.
- Guardrails 0.24.1 (Colang 1.0): each stage runs in its own `LLMRails` built from its own `.co` file;
  loading all flows together made flows of other stages fire. Retrieval rails only fire inside Guardrails'
  own knowledge-base loop, so the passage rail runs as an output flow on each passage.
- `langchain-community` is pinned to 0.4.1: `ragas` 0.4.3 fails to import with 0.4.2.
- `instructor` is pinned to 1.12.0: RAGAS leaves it unpinned and pip picked 0.4.0, which lacks `from_openai`.
- LangChain and LangGraph are installed only as dependencies of RAGAS and Guardrails. Our code does not use them
  (enforced by `tests/test_no_other_orchestration.py`).
- The toolkit's RAGAS evaluator is not used: it needs a LangChain-wrapped judge and cannot pass the embeddings
  that `AnswerRelevancy` requires. `eval/ragas_eval.py` calls RAGAS 0.4 directly.
- Web fact check: get a free key at https://app.tavily.com and set `TAVILY_API_KEY` in `.env`. Each claim checked
  costs 1 credit (`search_depth: basic`); results are cached in `data/cache/tavily_cache.sqlite`.
