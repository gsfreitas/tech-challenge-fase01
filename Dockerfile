# Dockerfile - ChurnAPI

FROM python:3.12-slim AS runtime

# variaveis de ambiente python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

# dependências de sistemas mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# instala gerenciador de pacotes uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# camada 1: dependencias
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# layer 2: codigo
COPY src/ ./src/

# layer 3: artefatos do modelo
COPY mlflow.db ./mlflow.db
COPY mlruns/ ./mlruns/
COPY models/ ./models/

# seguranca
# RUN groupadd --gid 1000 appgroup \
#     && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser \
#     && chown -R appuser:appgroup /app

# USER appuser

# healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

EXPOSE 8000

# entrypoint
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]