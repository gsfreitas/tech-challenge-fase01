# =============================================================
# Tech Challenge Fase 01 — Makefile
# =============================================================

.PHONY: help install lint format test test-fast test-cov train train-mlp \
        compare analyze api mlflow clean all

# Default: mostra ajuda
.DEFAULT_GOAL := help

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff

help:
	@echo "Tech Challenge Fase 01 - Make targets:"
	@echo ""
	@echo "  install       Instala dependencias com uv"
	@echo "  lint          Roda ruff check (sem auto-fix)"
	@echo "  format        Aplica ruff format e auto-fix"
	@echo "  test          Roda pytest com cobertura"
	@echo "  test-fast     Roda apenas unit tests"
	@echo "  test-cov      Roda testes e gera relatorio HTML em htmlcov/"
	@echo "  train         Treina baselines + MLP (sequencial)"
	@echo "  train-mlp     Treina apenas o MLP"
	@echo "  compare       Gera tabela comparativa de modelos"
	@echo "  analyze       Gera analise de custo FP vs FN"
	@echo "  api           Sobe API local em http://localhost:8000"
	@echo "  mlflow        Sobe MLflow UI em http://localhost:5000"
	@echo ""
	@echo "  docker-build  Builda imagem Docker (~2 min)"
	@echo "  docker-run    Roda container (porta 8000)"
	@echo "  docker-shell  Abre shell dentro do container (debug)"
	@echo "  docker-clean  Remove container e imagem"
	@echo ""
	@echo "  clean         Remove caches e arquivos temporarios"
	@echo "  all           install + lint + test"

# ─── Setup ───────────────────────────────────────────────────
install:
	uv sync --all-extras

# ─── Code quality ────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) check src/ tests/ --fix
	$(RUFF) format src/ tests/

# ─── Tests ───────────────────────────────────────────────────
test:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing

test-fast:
	$(PYTEST) tests/unit/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Cobertura HTML gerada em htmlcov/index.html"

# ─── Training ────────────────────────────────────────────────
train:
	$(PYTHON) -m src.training.train
	$(PYTHON) -m src.training.train_mlp

train-mlp:
	$(PYTHON) -m src.training.train_mlp

# ─── Analysis ────────────────────────────────────────────────
compare:
	$(PYTHON) -m src.training.compare_models

analyze:
	$(PYTHON) -m src.analysis.cost_analysis

# ─── Servers ─────────────────────────────────────────────────
api:
	@echo "Iniciando API com PYTHONPATH=src:."
	PYTHONPATH=src:. uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# -------------------------------------------------------------
# DOCKER
# -------------------------------------------------------------
# Variáveis (override: make docker-build IMAGE_TAG=v2)
IMAGE_NAME ?= churn-api
IMAGE_TAG ?= latest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo ""
	@echo "Imagem criada: $(IMAGE_NAME):$(IMAGE_TAG)"
	@docker images $(IMAGE_NAME):$(IMAGE_TAG) --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

docker-run:
	docker run --rm -p 8000:8000 \
		-e JWT_SECRET=dev-secret \
		-e ADMIN_PASSWORD=admin123 \
		-e USER_PASSWORD=user123 \
		--name $(IMAGE_NAME) \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-shell:
	docker run --rm -it --entrypoint /bin/bash $(IMAGE_NAME):$(IMAGE_TAG)

docker-clean:
	-docker stop $(IMAGE_NAME) 2>/dev/null
	-docker rm $(IMAGE_NAME) 2>/dev/null
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null
	@echo "Recursos Docker limpos."

# ─── Cleanup ─────────────────────────────────────────────────
clean:
	@echo "Removendo caches..."
	@rm -rf .pytest_cache .ruff_cache .coverage htmlcov 2>/dev/null || true
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Caches removidos."

# ─── Combos ──────────────────────────────────────────────────
all: install lint test