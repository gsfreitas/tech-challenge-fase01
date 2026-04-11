# 📉 Customer Churn Prediction — Tech Challenge Fase 01

> Projeto de Machine Learning para predição de churn em operadora de telecomunicações.  
> Desenvolvido como parte do programa de pós-graduação em Machine Learning Engineering.

---

## 🎯 Objetivo

Construir um pipeline end-to-end de predição de churn (cancelamento de clientes) utilizando dados públicos de telecomunicações, cobrindo desde a análise exploratória até a exposição do modelo via API de inferência.

---

## 📁 Estrutura do Projeto

```
tech-challenge-fase01/
│
├── data/ # Dados do projeto (não versionados)
│ ├── raw/ # Dados brutos originais (ex: CSV original)
│ └── processed/ # Dados tratados e prontos para modelagem
│
├── notebooks/ # Notebooks para exploração e análise
│ ├── exploratory/ # Análises exploratórias (EDA)
│ │ └── 01_eda.ipynb # Análise inicial dos dados
│ └── reports/ # Notebooks para apresentação
│
├── src/ # Código-fonte principal (produção)
│ ├── api/ # API de inferência (FastAPI)
│ │ └── init.py
│ │
│ ├── data/ # Scripts de ingestão e preparação de dados
│ │ └── init.py
│ │
│ ├── features/ # Engenharia de features
│ │ └── init.py
│ │
│ ├── models/ # Definição e registro de modelos
│ │ ├── init.py
│ │ └── register.py # Funções para salvar/carregar modelos
│ │
│ ├── training/ # Scripts de treinamento
│ │ ├── init.py
│ │ └── train.py # Pipeline de treino (baseline + MLP)
│ │
│ └── utils/ # Funções utilitárias gerais
│ ├── init.py
│ └── utils.py # Funções auxiliares (ex: logging, seeds, helpers)
│
├── tests/ # Testes automatizados
│ └── test_basic.py # Testes iniciais (smoke test / validações básicas)
│
├── docs/ # Documentação do projeto
│ # (ML Canvas, Model Card, arquitetura, etc.)
│
├── models/ # Modelos treinados (não versionados)
│ # (ex: .pkl, .pt)
│
├── mlruns/ # Experimentos do MLflow (não versionado)
│
├── LICENSE
└── README.md
```

---

## 📦 Datasets

Os dados **não estão incluídos** no repositório. Faça o download manualmente:

| Dataset | Fonte | Link |
|--------|-------|------|
| IBM Telco Customer Churn | Kaggle | [Download](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Iranian Churn Dataset | UCI ML Repository | [Download](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) |

Após o download, coloque os arquivos em `data/raw/`.

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/) para gerenciamento de dependências
- [UV](https://docs.astral.sh/uv/) para gerenciamento de dependências. 
- Artigo sobre UV: [UV Python Gerenciador de Pacotes Comparativo C#.](https://zocate.li/posts/2026/uv-python-gerenciador-pacotes-comparativo-csharp/)
- Docker (opcional, para containerização da API)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/tech-challenge-fase01.git
cd tech-challenge-fase01

# Instale as dependências
poetry install

# Ative o ambiente virtual
poetry shell
```

### Executando os Notebooks

```bash
jupyter notebook notebooks/
```

### Rastreamento de Experimentos (MLflow)

```bash
mlflow ui
# Acesse: http://localhost:5000
```

### API de Inferência

```bash
uvicorn src.api.main:app --reload
# Docs: http://localhost:8000/docs
```

---

## 🧪 Testes

```bash
pytest tests/ -v
```

---

## 🛠️ Stack

| Ferramenta | Uso |
|-----------|-----|
| `scikit-learn` | Modelagem e pré-processamento |
| `pandas` / `numpy` | Manipulação de dados |
| `matplotlib` / `seaborn` | Visualização |
| `mlflow` | Rastreamento de experimentos |
| `FastAPI` | API de inferência |
| `pytest` | Testes automatizados |
| `Docker` | Containerização |
| `Poetry` | Gerenciamento de dependências |

---

## 📊 Resultados

> _Seção a ser preenchida após a conclusão dos experimentos._

| Modelo | Acurácia | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Baseline (LR) | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

---

## 📝 Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.

---

## 👤 Autor

**Gabriel**  
Senior DevOps Engineer @ Telefônica Brasil  
Pós-graduando em Machine Learning Engineering