# 🧩 ML Canvas — Customer Churn Prediction

> Documento de alinhamento entre objetivos de negócio, dados, modelagem e operação do modelo preditivo de churn.
> Baseado no framework ML Canvas (Louis Dorard) adaptado ao contexto da Fase 1 do Tech Challenge POSTECH.

---

## 1. Proposta de Valor (Value Proposition)

**Problema de negócio:** uma operadora de telecomunicações enfrenta alta taxa de cancelamento (churn), com impacto direto em receita recorrente (MRR) e custo de aquisição de clientes (CAC).

**Proposta:** disponibilizar aos times de Retenção e CRM uma **probabilidade individual de churn** para cada cliente da base ativa, permitindo priorizar ações de retenção (ligação, desconto, upgrade de plano) nos clientes de maior risco.

**Ganho esperado:** reduzir churn evitável entre os clientes corretamente priorizados, otimizando orçamento de retenção ao concentrá-lo onde há maior probabilidade de cancelamento iminente.

---

## 2. Stakeholders & Usuários Finais

| Papel | Interesse | Como consome o modelo |
|---|---|---|
| Diretoria de Experiência do Cliente | KPI de churn mensal, ROI da operação | Dashboard agregado (top-N clientes em risco) |
| Time de Retenção (Call Center) | Lista priorizada de clientes para abordagem | Fila de contatos ordenada por probabilidade de Churn |
| Time de CRM / Marketing | Segmentação para campanhas de reengajamento | Scores exportados para ferramenta de CRM |
| Time de Dados (owner do modelo) | Performance, drift, explicabilidade | MLflow tracking + monitoramento |

---

## 3. Fontes de Dados

- **Dataset primário:** IBM Telco Customer Churn (Kaggle) — 7.043 clientes, 21 features, target binário (`Churn`).
- **Natureza:** tabular, snapshot estático — não há componente temporal explícito além de `tenure` (meses de contrato).
- **Tipos de variáveis:**
  - Demográficas: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Contratuais: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Serviços: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Financeiras: `MonthlyCharges`, `TotalCharges`
- **Qualidade conhecida:** 11 valores nulos em `TotalCharges` (clientes com `tenure=0`), sem duplicatas, classes desbalanceadas (~26,5% churn).
- **Versionamento:** dataset armazenado em `data/raw/`, processado em `data/processed/`, ambos fora do Git.

---

## 4. Previsão (Prediction Task)

- **Tipo de tarefa:** classificação binária supervisionada.
- **Variável alvo:** `Churn` → codificada como `1` (Yes, cancelou) ou `0` (No, permaneceu).
- **Saída do modelo:** probabilidade contínua ∈ [0, 1] de churn, com threshold configurável (default `0.5`, ajustado via análise de custo).
- **Granularidade:** um score por cliente (`customerID`).

---

## 5. Decisões (Decisions)

O score do modelo alimenta decisões de **priorização de ações de retenção**:

| Faixa de probabilidade | Ação recomendada |
|---|---|
| ≥ 0.70 | Contato ativo prioritário + oferta de retenção |
| 0.40 – 0.70 | Campanha de reengajamento segmentada (email/push) |
| < 0.40 | Sem ação imediata (manter monitoramento) |

Decisões operacionais finais permanecem com agentes humanos — o modelo **prioriza**, não **decide** descontos ou ofertas automaticamente ou toma ações de forma proativa.

---

## 6. Modelagem (Making Predictions)

- **Modelo central:** rede neural MLP (PyTorch) — definido pelo desafio.
- **Modelos baseline (comparação obrigatória):**
  - `DummyClassifier(strategy='most_frequent')` — sanity check
  - `LogisticRegression(class_weight='balanced')` — baseline linear interpretável
  - `DecisionTreeClassifier` — baseline não-linear interpretável
- **Pré-processamento:** pipeline sklearn com `ColumnTransformer` (imputação mediana/moda, `StandardScaler` para numéricas, `OneHotEncoder` para categóricas).
- **Validação:** `StratifiedKFold(n_splits=5, shuffle=True)` para CV + holdout 20% para avaliação final.
- **Reprodutibilidade:** `random_state=42` em todos os componentes estocásticos + `set_global_seed()` para `random`, `numpy`, `torch`.
- **Rastreamento:** MLflow — parâmetros, métricas CV, métricas holdout, artefatos (pipeline `.pkl`, `model.pt`).

---

## 7. Métricas

### Técnicas (offline)

**Primária:** **ROC-AUC** — métrica de ranqueamento robusta a desbalanceamento; alinhada ao uso do modelo (priorizar clientes).

**Secundárias:**
- **PR-AUC** — mais sensível ao desempenho na classe minoritária (churners).
- **F1-score** — harmônica entre precision e recall.
- **Recall (sensibilidade)** — % de churners reais capturados; crítico dado o custo alto de falso negativo.
- **Precision** — % de acerto entre clientes flagados; controla custo operacional.

### Negócio

- **Custo de churn evitado** = (nº de churners corretamente identificados × receita mensal média × horizonte de retenção) − (nº de abordagens × custo unitário de retenção).
- **Lift @ top decil** — quantas vezes mais churners o modelo identifica no top 10% vs. distribuição aleatória.
- **Taxa de conversão de retenção** — % de clientes flagados que permaneceram após ação (KPI operacional, medido pós-deploy).

### SLOs (Service Level Objectives)

- **Latência de inferência:** p95 < 200ms por requisição na API.
- **Disponibilidade:** 99% uptime (ambiente de homologação; produção definido em fase de deploy).
- **Freshness:** modelo retreinado a cada 3 meses ou quando drift detectado (PSI > 0.2).

---

## 8. Avaliação (Evaluation)

### Offline
- Cross-validation estratificada (5 folds) em todos os modelos.
- Holdout 20% estratificado para métricas finais reportadas.
- Comparação MLP vs. baselines em tabela única com todas as métricas + intervalo de confiança (desvio padrão dos folds).
- Análise de **trade-off custo FP vs. FN**: varredura de thresholds (0.1 → 0.9) com curva de custo esperado, definindo threshold operacional.

### Online (pós-deploy, fora do escopo da Fase 1)
- A/B test: grupo de clientes priorizados pelo modelo vs. grupo priorizado por regra heurística atual.
- KPIs comparados: churn real em 30/60/90 dias, ROI de retenção.

---

## 9. Monitoramento & Riscos

### Monitoramento
- **Performance drift:** ROC-AUC, PR-AUC e recall calculados mensalmente em amostra rotulada (janela móvel de 60 dias).
- **Data drift:** PSI (Population Stability Index) nas top-10 features mais importantes; alerta se PSI > 0.2.
- **Concept drift:** comparação de distribuição de `y_pred` entre treino e produção.
- **Alertas:** queda de ROC-AUC > 5 pp em relação ao baseline treino → trigger de retrain.

### Riscos identificados

| Risco | Impacto | Mitigação |
|---|---|---|
| **Falsos negativos** (churners não detectados) | Perda direta de receita | Otimizar threshold priorizando recall; custo FN > custo FP na função de decisão |
| **Falsos positivos** (clientes estáveis flagados) | Custo operacional de retenção desperdiçado + risco de incomodar cliente com ofertas desnecessárias | Controlar precision no top decil; limitar volume diário de contatos |
| **Desbalanceamento de classes** (~26% churn) | Modelo enviesado para classe majoritária | `class_weight='balanced'` nos baselines; `pos_weight` no BCELoss do MLP; avaliar com PR-AUC |
| **Data leakage no pré-processamento** | Métricas offline otimistas, degradação em produção | Toda transformação estatística (imputer, scaler, encoder) dentro do `Pipeline` sklearn, fitada apenas no fold de treino |
| **Vieses demográficos** | Decisões discriminatórias (gênero, idade) | Avaliação de fairness segmentada por `gender` e `SeniorCitizen` no Model Card |
| **Deriva de comportamento** pós-promoção ou mudança de pricing | Modelo desatualiza rápido | Monitoramento de drift + retrain trigger automático |
| **Dataset público vs. realidade** | Dataset IBM não reflete 100% a operação real; generalização limitada | Documentar limitação explícita no Model Card; tratar como PoC acadêmico |
