FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[openai,mlops]" || pip install --no-cache-dir \
    pydantic pydantic-settings qdrant-client fastapi uvicorn

COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src:/app

EXPOSE 8000
CMD ["uvicorn", "ragkit.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
