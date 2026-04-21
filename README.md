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
│ ├── config.py # Paths e configurações globais
│ ├── reproducibility.py # Controle de seeds
│ └── logging_config.py # Configuração de logging
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

## 🔄 Fluxo do Projeto

O projeto segue uma abordagem modular, separando responsabilidades entre exploração, processamento e modelagem:

1. **Ingestão de dados**
   - Leitura do dataset bruto em `data/raw/`
   - Centralizada em `src/data/data_loader.py`

2. **Limpeza e preparação**
   - Tratamento inicial de dados
   - Implementado em `src/data/data_cleaner.py`

3. **Exploração (EDA)**
   - Análises exploratórias nos notebooks
   - `notebooks/exploratory/`

4. **Configuração e reprodutibilidade**
   - Paths e constantes centralizados em `src/utils/config.py`
   - Seed global definida em `src/utils/reproducibility.py`

5. **Modelagem (em evolução)**
   - Baselines e MLP serão implementados nas próximas etapas

6. **API de inferência (futuro)**
   - Será implementada com FastAPI em `src/api/`

---

### Configuração de ambiente e reprodutibilidade

O projeto utiliza uma configuração centralizada para garantir consistência entre ambientes:

- Paths definidos em `src/utils/config.py`
- Seed global definida em `src/utils/reproducibility.py`

Isso garante que experimentos sejam reproduzíveis.

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

Os experimentos de modelagem são rastreados utilizando MLflow, permitindo:

- registro de métricas
- comparação entre modelos
- versionamento de experimentos

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

### Comparativo de modelos (holdout test set)

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| Dummy (baseline) | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic Regression | 0.8431 | 0.6385 | 0.6157 | 0.7861 | 0.5060 | 0.7395 |
| Decision Tree | 0.7939 | 0.5659 | 0.5962 | 0.7914 | 0.4782 | 0.7154 |
| **MLP (PyTorch)** | **0.8470** | 0.6318 | 0.6197 | 0.7857 | 0.5116 | 0.7446 |

### Análise de custo FP vs FN

Assumindo custos assimétricos de negócio — **R$ 1.500 por Falso Negativo** (receita perdida ao deixar um churner passar) vs **R$ 50 por Falso Positivo** (custo de uma abordagem proativa desnecessária) — a análise de threshold mostra que o threshold padrão (0.5) é sub-ótimo:

| Configuração | Threshold | FP | FN | Recall | Custo total |
|---|---|---|---|---|---|
| Padrão | 0.50 | 210 | 60 | 78.6% | R$ 100.500 |
| **Ótimo** | **0.11** | 531 | 0 | 100.0% | R$ 26.550 |

**Redução de custo ao usar threshold ótimo: 73,6%.**

### Key insights

- **MLP e LogReg empatam tecnicamente.** Para dados tabulares de ~7k linhas, a complexidade da rede neural não traz ganho proporcional sobre um baseline linear bem configurado. Esse é um padrão conhecido na literatura — MLPs brilham mais em dados não-tabulares (imagem, texto, áudio) ou quando há volume muito maior de exemplos.
- **O modelo a ser servido é o MLP**, por ser o modelo central do desafio, com threshold operacional de 0.11 (não o padrão 0.5).
- **Recall é a métrica crítica** para este caso de uso. O custo de perder um churner é 30x maior que o custo de abordar um cliente estável.

Artefatos detalhados em `docs/model_comparison.csv`, `docs/cost_analysis.csv` e gráficos em `docs/*.png`. Notebook consolidado em `notebooks/reports/04_model_comparison.ipynb`.

---


## 📚 Documentação

A documentação do projeto está disponível em `docs/`:

- `ml_canvas.md` → definição do problema e estratégia de ML
- `problem_definition.md` → detalhamento técnico do problema
- (futuro) `model_card.md` → descrição do modelo e limitações

---

## 📝 Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.

---

## 👤 Autors

**Gabriel**
Senior DevOps Engineer @ Telefônica Brasil

**Diego**
Full Stack Developer @ Eldorado Research Institute