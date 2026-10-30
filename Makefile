# Needs make + Docker. On Windows without them, use .\demo.ps1 instead.
.PHONY: demo api ui test examples

demo:  ## API + UI in Docker: http://localhost:8501
	docker compose -f docker/docker-compose.yml up --build

api:
	uvicorn src.api.app:app --port 8000

ui:
	streamlit run src/ui/app.py

test:
	python -m pytest -q

examples:  ## precompute the UI's cached examples (makes live API calls once)
	python scripts/make_examples.py
