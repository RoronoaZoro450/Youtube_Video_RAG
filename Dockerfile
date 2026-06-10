FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY api.py .
COPY Ingestion/ ./Ingestion/
COPY Pipeline/ ./Pipeline/
COPY Retrieval/ ./Retrieval/

# Expose port
EXPOSE 8000

# Run
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]