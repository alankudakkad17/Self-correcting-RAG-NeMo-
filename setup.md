# Setup (Windows, native Python)

Tested with Python 3.13.14 (the toolkit needs >=3.11,<3.14). On this machine `py -3.11` resolves to
Anaconda 3.10, so use `py -3.13`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps   # registers src/register.py as a toolkit plugin
copy .env.example .env                                      # then put your key in .env
```

Behind Kaspersky HTTPS scanning, add to every pip command:
`--trusted-host pypi.org --trusted-host files.pythonhosted.org`

## Milestone 1 checks

```powershell
# Offline unit tests (no API calls)
.\.venv\Scripts\python.exe -m pytest -q

# Config is valid and the plugin is found
.\.venv\Scripts\nat.exe validate --config_file configs/smoke.yml

# One call per configured model (generator, verifiers, judges); results -> eval/results/smoke_models.json
.\.venv\Scripts\python.exe scripts/smoke_models.py

# Minimal toolkit workflow end to end
.\.venv\Scripts\nat.exe run --config_file configs/smoke.yml --input "What is the capital of France?"
```

Running a command a second time is served from `data/cache/llm_cache.sqlite` and uses no quota.
Set `LLM_CACHE_ENABLED=0` to bypass the cache (needed for honest latency numbers).

## Milestone 2: data, indices, baselines, evaluation

```powershell
# 1. Download pinned sources (HotpotQA distractor validation, 27 MB; MiniLM reranker, 91 MB) into data/
.\.venv\Scripts\python.exe scripts/download_data.py

# 2. Build the frozen corpus + dev/test/pilot sets (refuses to overwrite; the committed files are the frozen v1)
.\.venv\Scripts\python.exe scripts/make_eval_sets.py

# 3. Embed the corpus and build the FAISS index (~80 s, ~112 embedding calls, cached)
.\.venv\Scripts\python.exe scripts/build_index.py

# 4. Run a system on a split (pilot = 20 dev items; never tune on test)
.\.venv\Scripts\python.exe eval/run_eval.py --config configs/B.yml --split pilot

# 5. RAGAS scores for that run (judge = `judge` in configs/llm.yml)
.\.venv\Scripts\python.exe eval/ragas_eval.py --run eval/results/runs/<run_dir>
```

Systems: `configs/H.yml` (LLM only), `configs/A.yml` (dense top-5), `configs/B.yml`
(BM25 + dense -> RRF -> hosted Nemotron rerank -> top-5). Try one question:
`.\.venv\Scripts\nat.exe run --config_file configs/B.yml --input "<question>"`

## Milestone 3: Planner, Researcher, Verifier, Controller (systems C, D)

```powershell
# Controller rules and claim parsing, with scripted fake agents (no API calls)
.\.venv\Scripts\python.exe -m pytest tests/test_controller.py tests/test_parsing.py -q

# One question through D; the JSON output includes the plan, claims with verdicts and the per-attempt trace
.\.venv\Scripts\nat.exe run --config_file configs/D.yml --input "<question>"

# Pilot (20 dev items)
.\.venv\Scripts\python.exe eval/run_eval.py --config configs/D.yml --split pilot
```

`configs/D.yml` inherits `configs/C.yml` (`base:`) and adds the Verifier and `retry_cap` (default 2).

## Known version workarounds
- `nvidia-nat[guardrails]` is not used: it pins `nemoguardrails<0.22`, which needs `annoy`
  (no Windows wheel). We install `nemoguardrails==0.24.1` on its own.
- `langchain-community` is pinned to 0.4.1: `ragas` 0.4.3 fails to import with 0.4.2.
- `instructor` is pinned to 1.12.0: RAGAS leaves it unpinned and pip picked 0.4.0, which lacks `from_openai`.
- LangChain and LangGraph are installed only as dependencies of RAGAS and Guardrails. Our code does not use them
  (enforced by `tests/test_no_other_orchestration.py`).
- The toolkit's RAGAS evaluator is not used: it needs a LangChain-wrapped judge and cannot pass the embeddings
  that `AnswerRelevancy` requires. `eval/ragas_eval.py` calls RAGAS 0.4 directly.
