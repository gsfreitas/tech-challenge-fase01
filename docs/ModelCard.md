# Model Card — Churn Prediction MLP

> Formato adotado: **Model Cards for Model Reporting**
> Última atualização: 2026-04-26
> Versão do modelo: **2.1.0**
> Estágio MLflow: `Production`

---

## 1. Model Details

Informações básicas sobre o modelo.

| Atributo | Valor |
|---|---|
| **Nome** | `churn-mlp-pytorch` |
| **Tipo** | Multi-Layer Perceptron (rede neural feedforward) |
| **Versão** | 2.1.0 |
| **Data do treino** | [FILL: data do último treino do MLP, ex: 2026-04-20] |
| **Framework** | PyTorch 2.x + scikit-learn 1.x |
| **Autor(es)** | Grupo 2 — 9MLET FIAP ([FILL: nomes dos integrantes]) |
| **Repositório** | https://github.com/gsfreitas/tech-challenge-fase01 |
| **Licença** | MIT |
| **Tipo de problema** | Classificação binária supervisionada |
| **Saída** | Probabilidade ∈ [0, 1] de churn + decisão binária aplicando threshold operacional 0.11 |
| **Paper de referência** | Modelo customizado; arquitetura inspirada em práticas standard para tabular data |

### Arquitetura

```
Input (60 features após preprocessing)
  → Linear(60 → 128) → ReLU → Dropout(0.3)
  → Linear(128 → 64) → ReLU → Dropout(0.3)
  → Linear(64 → 32)  → ReLU → Dropout(0.2)
  → Linear(32 → 1)   → sigmoid (na inferência)

Total de parâmetros treináveis: 18.177
```

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | BCEWithLogitsLoss (com `pos_weight=2.77`) |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 (monitora `val_loss`) |
| Random seed | 42 |
| Best epoch (treino atual) | 4 |

---

## 2. Intended Use

Para que o modelo foi construído, e o que ele **não** deveria fazer.

### Uso primário

- **Caso de uso:** identificar clientes de uma operadora de telecom com alta probabilidade de cancelamento (churn) nos próximos meses, para priorização de ações de retenção.
- **Modo de operação esperado:** **batch** — execução periódica sobre a base ativa de clientes, gerando lista priorizada.
- **Usuários previstos:**
  - Time de Retenção / Customer Success
  - Diretoria comercial
  - Engenheiros de ML

### Usos *fora do escopo* (out-of-scope)

O modelo **não deve ser usado** para:

- Decisões de **preço dinâmico** ou ajuste de tarifas sem revisão humana — não foi otimizado para isso.
- **Negar serviço** a um cliente baseado em risco de churn previsto.
- Predições para clientes **fora do contexto IBM Telco** (clientes com perfis muito diferentes do dataset de treino).
- Análise individual em produção sem **revisão humana**.
- Substituir relatórios ou auditorias formais.

### Limitações de uso

- O modelo é **treinado em dados sintéticos públicos** (IBM Telco) que **não refletem 100%** o perfil de uma operadora específica.
- Threshold operacional **0.11** foi calibrado para a estrutura de custo simulada (FN = R$1.500, FP = R$50). Em outro contexto de negócio, o threshold deve ser **recalibrado**.

---

## 3. Fatores

Fatores relevantes que podem influenciar a performance do modelo.

### Fatores demográficos presentes nos dados

| Fator | Valores | Distribuição |
|---|---|---|
| `gender` | Male / Female | ~50% / ~50% |
| `SeniorCitizen` | 0 / 1 | ~84% / ~16% |
| `Partner` | Yes / No | ~48% / ~52% |
| `Dependents` | Yes / No | ~30% / ~70% |

### Fatores instrumentais (relacionados ao serviço)

- **Tipo de contrato** (`Contract`): Month-to-month / One year / Two year
- **Método de pagamento** (`PaymentMethod`): Electronic check / Mailed check / Bank transfer / Credit card
- **Internet** (`InternetService`): DSL / Fiber optic / No
- **Tempo de relacionamento** (`tenure`): 0 a 72 meses

### Fatores de ambiente *não* representados no dataset

Os seguintes fatores **não foram considerados** e podem afetar performance em produção real:

- Localização geográfica do cliente
- Histórico de atendimento e reclamações
- Histórico de pagamentos atrasados

---

## 4. Metrics

Métricas escolhidas, motivações e variação esperada.

### Métricas relatadas

| Métrica | Valor (holdout) | Por que importa |
|---|---|---|
| **ROC-AUC** | **0.8470** | Métrica primária — captura discriminação geral, robusta a desbalanceamento |
| **PR-AUC** | 0.6318 | Foco na classe minoritária (churners), mais informativa que ROC para alvos raros |
| **Recall (sensibilidade)** | 0.7857 | Crítica — perder um churner custa 30× mais que falso alarme |
| **Precision** | 0.5116 | Indica qualidade da lista de retenção (taxa de "alarmes verdadeiros") |
| **F1** | 0.6197 | Equilíbrio entre precision e recall |
| **Accuracy** | 0.7446 | Reportada para comparabilidade, mas pouco informativa em dataset desbalanceado (26% churn) |

### Threshold de decisão

Threshold padrão de 0.5 produziria custo total de **R\$ 100.500** no test set.
Threshold operacional adotado: **0.11**, que produz custo de **R\$ 26.550** — redução de **73,6%**.

Detalhes em `docs/cost_analysis.csv` e `notebooks/reports/04_compara_modelos.ipynb`.

### Variação esperada

- Reprodutibilidade: com seed `42` e mesma versão de bibliotecas, métricas reproduzem com diferença < 0.001.

### Estratégia de decisão

- `proba > 0.11` → cliente entra na lista de retenção
- `proba ≤ 0.11` → cliente não é flagado nesta rodada

A decisão final sobre **qual ação tomar** com cada cliente flagado é sempre **com revisão humana**.

---

## 5. Evaluation Data

Dados usados para avaliar o modelo final.

### Dataset

- **Origem:** IBM Telco Customer Churn (público, disponível no Kaggle)
- **Tamanho total:** 7.043 clientes, 21 features brutas
- **Holdout:** 1.057 clientes (15% do total) — separado **antes** do treino, nunca tocado durante seleção de modelo
- **Distribuição de classes no holdout:** 73,5% não-churn / 26,5% churn (estratificada)

### Pré-processamento

1. Remoção de duplicatas e tratamento de `TotalCharges` com strings inválidas
2. Imputação de medianas (apenas com base no train set, sem leakage)
3. Feature engineering: 7 features derivadas (`FamilyStatus`, `TenureBin`, `RiskProfile`, `RiskCombo`, `SupportBundle`, `DiffMonthlyCharges`, `FamilySizeProxy`)
4. One-hot encoding de categóricas + StandardScaler em numéricas
5. Total final: **60 features** após preprocessing

---

## 6. Training Data

Dados usados para treinar o modelo.

### Composição

- **Total de exemplos no treino:** 4.929 clientes (70%)
- **Total no validation set:** 1.057 clientes (15%, usado para early stopping)
- **Distribuição de classes (treino):** 73,5% não-churn / 26,5% churn
- **Estratificação:** o split foi estratificado por `Churn` para preservar proporção de classes nos 3 conjuntos.


### Class weighting

Para mitigar desbalanceamento, foi aplicado `pos_weight = 2.77` no `BCEWithLogitsLoss`, calculado como `n_negativos / n_positivos`. Equivalente ao `class_weight='balanced'` do scikit-learn para penalizar a classe minoritária (neste caso, clientes com churn).

---

## 7. Quantitative Analyses

Análise desagregada de performance — onde o modelo brilha e onde falha.

### Performance comparativa (holdout)

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| Dummy (baseline) | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic Regression | 0.8431 | 0.6385 | 0.6157 | 0.7861 | 0.5060 | 0.7395 |
| Decision Tree | 0.7939 | 0.5659 | 0.5962 | 0.7914 | 0.4782 | 0.7154 |
| **MLP (PyTorch)** | **0.8470** | 0.6318 | 0.6197 | 0.7857 | 0.5116 | 0.7446 |

### Insight central

**MLP e Logistic Regression empatam tecnicamente** (diferença de 0.4 pp em ROC-AUC, dentro do ruído estatístico). Para os dados tabulares com sinais predominantemente lineares, a complexidade adicional de uma rede neural não trouxe ganho proporcional.

A LogReg, sendo **mais simples e interpretável**, é uma alternativa válida em produção.

### Performance por subgrupo

| Subgrupo            | n | ROC-AUC      | Recall | Precision |
| ------------------- | - | ------------ | ------ | --------- |
| `gender = Female`   | — | ~0.845       | 1.000  | ~0.340    |
| `gender = Male`     | — | ~0.849       | 1.000  | ~0.350    |
| `SeniorCitizen = 0` | — | ~0.807–0.887 | 1.000  | ~0.270    |
| `SeniorCitizen = 1` | — | ~0.807–0.887 | 1.000  | ~0.417    |


### Análise de erros

- **Falsos negativos com threshold 0.11:** 0 (todos os churners do holdout foram capturados)
- **Falsos positivos com threshold 0.11:** 531 — clientes não-churners flagados para retenção
- **Cobertura da classe positiva:** 100% (recall = 1.0 quando threshold = 0.11)

---

## 8. Considerações Éticas

Riscos éticos identificados e mitigações propostas.

### Risco 1 — Vieses demográficos não-mitigados

**Descrição:** o modelo usa variáveis demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) que podem introduzir tratamento desigual entre grupos.

**Mitigação atual:** análise descritiva de fairness foi conduzida (ver seção 7 e notebook `05_fairness_analysis.ipynb`). Disparidades foram documentadas, mas **não foram aplicadas técnicas de mitigação** (ex: `Fairlearn` reweighing).

**Mitigação recomendada para produção:**
- Aplicar `Fairlearn` para mensurar e mitigar disparidades formalmente
- Estabelecer SLA de fairness com stakeholders (ex: "diferença de recall entre subgrupos < 5pp")
- Monitorar continuamente em produção


### Risco 2 — Alta taxa de Falsos Positivos no threshold

**Descrição:** com threshold 0.11, ~50% dos clientes flagados são falsos positivos (pessoas que **não** iam churnar). Isso significa **incomodar metade** dos clientes da lista com ações de retenção desnecessárias.

**Mitigação:**
- Ações de retenção devem ser **bem desenhadas** e não invasivas
- Limite de quantas vezes um cliente pode ser flagado em janela de 6 meses
- Mecanismo de **opt-out** para clientes que não querem ser contatados
- Revisar threshold periodicamente conforme premissas de custo evoluam

---


### Recomendações

#### Curto prazo (próximos 30 dias após deploy hipotético)

- [ ] Implementar **monitoramento de drift** (PSI nas features chave: `tenure`, `MonthlyCharges`, `Contract`)
- [ ] Implementar **logging detalhado** de cada predição em produção para auditoria

#### Médio prazo (3-6 meses)

- [ ] **Retreinar** com dados internos da operadora
- [ ] Aplicar **Fairlearn** para mitigação formal de viés