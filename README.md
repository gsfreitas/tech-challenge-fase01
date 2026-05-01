# 📉 Customer Churn Prediction — Tech Challenge Fase 01

> Pipeline end-to-end de predição de churn em telecomunicações: do EDA ao deploy.
> Desenvolvido como parte do programa **Machine Learning Engineering** — PosTech FIAP.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-151%20passing-success.svg)](#-testes)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Objetivo

Construir um **pipeline completo de predição de churn** para uma operadora de telecomunicações, cobrindo:

- 📊 Análise exploratória e engenharia de features
- 🤖 Treinamento de modelos baseline (LogReg, Decision Tree) e MLP em PyTorch
- 📈 Análise de custo e calibração de threshold operacional
- 🔌 API de inferência com FastAPI, JWT e roles
- 🧪 Suíte de testes automatizada (151 testes)
- 📚 Documentação completa (Model Card, Monitoring Plan, Deployment Architecture)

---

## 🚀 Quick Start

```bash
# 1. Clone o repositório
git clone https://github.com/gsfreitas/tech-challenge-fase01.git
cd tech-challenge-fase01

# 2. Instale dependências
make install   # ou: uv sync --all-extras

# 3. Configure variáveis de ambiente
cp .env.example .env

# 4. Treine os modelos (gera mlflow.db com Registry populado)
make train     # ~1-2 minutos

# 5. Suba a API
make api       # http://localhost:8000/docs
```

Pra Windows sem Make: `$env:PYTHONPATH = "src;."; uv run uvicorn src.api.main:app --reload`

---

## 📁 Estrutura do Projeto

```
tech-challenge-fase01/
├── data/                          # Dados (não versionados)
│   ├── raw/                       # CSV original do IBM Telco
│   └── processed/                 # Dados após preprocessing
│
├── notebooks/
│   ├── exploratory/               # Análises iniciais (EDA)
│   │   ├── 01_eda.ipynb
│   │   ├── 02_feature_analysis.ipynb
│   │   └── 03_baseline_model.ipynb
│   └── reports/                   # Notebooks finais (entregáveis)
│       ├── 04_compara_modelos.ipynb
│       └── 05_fairness_analysis.ipynb
│
├── src/
│   ├── api/                       # API FastAPI
│   │   ├── core/                  # Auth (JWT), config
│   │   ├── middleware/            # Rate limit, latency
│   │   ├── models/                # Schemas Pydantic
│   │   ├── routes/                # auth, predict, system
│   │   ├── services/              # ModelService, feature engineering
│   │   └── main.py
│   │
│   ├── analysis/                  # Análise de custo FP/FN
│   ├── data/                      # Loaders e cleaners
│   ├── features/                  # Feature engineering + preprocessing
│   ├── models/                    # MLP PyTorch + early stopping
│   ├── training/                  # Scripts de treino e comparação
│   └── utils/                     # Config, reproducibility, logging
│
├── tests/
│   ├── unit/                      # 121 testes unitários
│   └── api/                       # 30 testes de API
│
├── docs/
│   ├── model_card.md              # Model Card (Mitchell et al. 2019)
│   ├── monitoring_plan.md         # Plano de monitoramento
│   ├── deployment_architecture.md # Arquitetura de deploy
│   ├── ml_canvas.md               # ML Canvas
│   └── (gráficos PNG, CSVs)
│
├── models/                        # Artefatos treinados (não versionados)
├── mlflow.db                      # SQLite com Model Registry (versionado)
├── Makefile                       # Atalhos comuns
├── pyproject.toml                 # Dependências e configurações
└── .env.example                   # Template de variáveis de ambiente
```

---

## 📊 Resultados

### Comparativo de modelos (holdout test set)

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| Dummy (baseline) | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic Regression | 0.8431 | 0.6385 | 0.6157 | 0.7861 | 0.5060 | 0.7395 |
| Decision Tree | 0.7939 | 0.5659 | 0.5962 | 0.7914 | 0.4782 | 0.7154 |
| **MLP (PyTorch)** | **0.8470** | 0.6318 | 0.6197 | 0.7857 | 0.5116 | 0.7446 |

### Análise de custo FP vs FN

Assumindo custos assimétricos de negócio — **R\$ 1.500 por Falso Negativo** (receita perdida) vs **R\$ 50 por Falso Positivo** (custo de abordagem desnecessária) — a análise de threshold revelou que o padrão (0.5) é sub-ótimo:

| Configuração | Threshold | FP | FN | Recall | Custo total |
|---|---|---|---|---|---|
| Padrão | 0.50 | 210 | 60 | 78.6% | R\$ 100.500 |
| **Operacional** | **0.11** | 531 | 0 | 100.0% | R\$ 26.550 |

**Redução de custo: 73,6%** ao usar o threshold operacional.

### Key insights

- **MLP e LogReg empatam tecnicamente.** Para dados tabulares de ~7k linhas, a complexidade adicional da rede não traz ganho proporcional.
- **Recall é a métrica crítica** — perder um churner custa 30× mais que abordar um cliente estável.
- **MLP escolhido como modelo principal** pelo escopo acadêmico (demonstrar arquitetura PyTorch end-to-end), com threshold operacional `0.11` configurado em `src/utils/config.py`.

Artefatos completos em `docs/model_comparison.csv`, `docs/cost_analysis.csv` e gráficos em `docs/*.png`. Notebooks consolidados em `notebooks/reports/`.

---

## 🔌 API de Inferência

API FastAPI com **autenticação JWT**, **rate limiting**, **logging estruturado JSON** e **middleware de latência**.

### Endpoints

| Endpoint | Método | Auth | Role | Descrição |
|---|---|---|---|---|
| `/` | GET | Não | — | Metadata da API |
| `/health` | GET | Não | — | Health check (verifica modelos carregados) |
| `/auth/login` | POST | Não | — | Gera JWT token |
| `/auth/me` | GET | Sim | qualquer | Dados do usuário autenticado |
| `/predict/mlp` | POST | Sim | **admin** | Predição com MLP (PyTorch) |
| `/predict/lr` | POST | Sim | qualquer | Predição com Logistic Regression |
| `/predict/tree` | POST | Sim | qualquer | Predição com Decision Tree |
| `/docs` | GET | Não | — | Swagger UI |

### Roles e usuários

| Role | Acesso |
|---|---|
| `admin` | Todos os endpoints, incluindo `/predict/mlp` |
| `user` | Apenas baselines (`/predict/lr`, `/predict/tree`) |

Usuários padrão configurados via `.env`. Em produção real, substituir por banco com hash bcrypt.

### Exemplo de uso (curl)

```bash
# 1. Login (obter JWT)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' \
  | jq -r '.access_token')

# 2. Predizer churn
curl -X POST http://localhost:8000/predict/mlp \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0
  }'
# Esperado: {"churn_prediction": 1, "churn_probability": 0.87}
```

### Postman

Collection pronta em [`docs/ChurnAPI.postman_collection`](docs/ChurnAPI.postman_collection)

---

## 🧠 Pipeline de ML

### MLflow Model Registry

Os 3 modelos são registrados automaticamente no MLflow Registry com stage `Production` ao final de cada treino:

- `logistic_regression` v1 → Production
- `decision_tree` v1 → Production
- `mlp_pytorch` v1 → Production

A API carrega os modelos via `models:/<n>/Production`, garantindo que sempre serve a versão promovida.

### Visualizar experimentos

```bash
make mlflow                          # http://localhost:5000
# ou: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Reprodutibilidade

- Seed global `42` em `src/utils/reproducibility.py`
- Split 70/15/15 estratificado por `Churn`
- Backend MLflow em SQLite (`mlflow.db`) versionado no repo
- Hiperparâmetros do MLP fixos (documentados no Model Card)

---

## 🧪 Testes

**151 testes** automatizados, organizados por escopo:

| Categoria | Testes | Cobertura |
|---|---|---|
| Unit (`tests/unit/`) | 121 | Data, features, modelos, training, analysis |
| API (`tests/api/`) | 30 | Health, auth, predict, validação |

```bash
make test          # com cobertura
make test-fast     # apenas unit (mais rápido)
make test-cov      # gera HTML em htmlcov/
```

**Linting com Ruff:**
```bash
make lint          # check apenas
make format        # auto-fix + format
```

---

## 🛠️ Stack

| Categoria | Ferramentas |
|---|---|
| **ML** | scikit-learn, PyTorch, NumPy, Pandas |
| **Tracking** | MLflow Registry (SQLite backend) |
| **API** | FastAPI, Pydantic v2, uvicorn |
| **Auth** | PyJWT, rate limiting custom |
| **Observability** | Logging JSON estruturado, middleware de latência |
| **Testing** | pytest, pytest-cov, pytest-mock, httpx |
| **Code quality** | Ruff (lint + format) |
| **Package mgmt** | uv |
| **Containerization** | Docker (deploy AWS) |

---

## 📚 Documentação

Documentação técnica completa em `docs/`:

| Documento | Descrição |
|---|---|
| [`ml_canvas.md`](docs/ml_canvas.md) | Definição do problema e estratégia de ML |
| [`problem_definition.md`](docs/problem_definition.md) | Detalhamento técnico do problema |
| [`model_card.md`](docs/model_card.md) | **Model Card** completo (Mitchell et al. 2019, 9 seções) |
| [`monitoring_plan.md`](docs/monitoring_plan.md) | **Plano de monitoramento** com PSI, alertas e playbooks |
| [`deployment_architecture.md`](docs/deployment_architecture.md) | **Arquitetura de deploy** (batch + API, AWS App Runner) |

Notebooks consolidados em `notebooks/reports/`:

- `04_compara_modelos.ipynb` — comparação detalhada dos 4 modelos
- `05_fairness_analysis.ipynb` — análise de viés por subgrupo demográfico

---

## 🐳 Deploy

A API foi projetada para deploy via Docker.

```bash
# Build da imagem Docker
docker build -t churn-api .

# Run local
docker run -p 8000:8000 \
  -e JWT_SECRET=your-secret \
  -e ADMIN_PASSWORD=admin123 \
  churn-api
```

---

## 📦 Datasets

Os dados **não estão versionados**. Faça download e coloque em `data/raw/`:

| Dataset | Fonte | Link |
|---|---|---|
| **IBM Telco Customer Churn** (usado) | Kaggle | [Download](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

**Sobre os modelos:** após `make train`, os artefatos `.pkl`/`.pt` ficam em `models/` (gitignored). Para rodar a API sem treinar, baixe a release com modelos pré-treinados:

```powershell
# Windows
.\scripts\download_model.ps1
```

---


## 🛠️ Comandos disponíveis (Makefile)

```bash
make help          # Lista todos os targets
make install       # Instala dependências com uv
make lint          # Roda ruff check
make format        # Aplica ruff format + auto-fix
make test          # Roda pytest com cobertura
make test-fast     # Apenas unit tests
make test-cov      # Gera relatório HTML em htmlcov/
make train         # Treina baselines + MLP
make compare       # Gera tabela comparativa
make analyze       # Gera análise de custo FP vs FN
make api           # Sobe API local em :8000
make mlflow        # Sobe MLflow UI em :5000
make clean         # Remove caches
```

---

## 📈 Pré-requisitos

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** para gerenciamento de dependências
- (Opcional) **Make** para atalhos
- (Opcional) **Docker** para containerização

> Sobre o uv: [UV Python Gerenciador de Pacotes — Comparativo](https://zocate.li/posts/2026/uv-python-gerenciador-pacotes-comparativo-csharp/)

---

## 📝 Licença

Distribuído sob a licença MIT. Consulte [LICENSE](LICENSE) para mais informações.

---

## 👤 Autores

**Grupo 2 — 9MLET FIAP**

| Nome | Cargo |
|---|---|
| **Gabriel Freitas** | Senior DevOps Engineer @ Telefônica Brasil |
| **Diego** | Full Stack Developer @ Eldorado Research Institute |
| **Deyvid** | Fullstack Developer @ Minsait |
| **Lucas Molitor** | Fullstack Developer @ CI&T |