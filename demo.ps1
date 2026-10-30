# Local demo without Docker or make (Windows): starts the API in a new window, then the UI here.
# API: http://localhost:8000   UI: http://localhost:8501
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "src.api.app:app", "--port", "8000" -WorkingDirectory $PSScriptRoot
& $py -m streamlit run (Join-Path $PSScriptRoot "src\ui\app.py")
