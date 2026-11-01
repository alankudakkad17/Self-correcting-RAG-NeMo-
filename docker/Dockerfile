# One image for both services (api, ui); docker-compose picks the command.
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# requirements.txt is a Windows lock: drop the Windows-only package before installing on Linux.
COPY requirements.txt .
RUN grep -v -i '^pywin32' requirements.txt > requirements-linux.txt \
    && pip install --no-cache-dir -r requirements-linux.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY eval ./eval
COPY scripts ./scripts
RUN pip install --no-cache-dir -e . --no-deps

# data/ (frozen sets, index, caches) is mounted at runtime, see docker-compose.yml
EXPOSE 8000 8501
