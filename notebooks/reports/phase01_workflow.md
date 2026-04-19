## Baseline Model — Processo

### Objetivo

Estabelecer um desempenho mínimo para o problema de churn, servindo como referência para modelos mais complexos.

---

### Etapas realizadas

1. Definição da variável target (`Churn → target`)
2. Separação de features e target
3. Identificação de colunas numéricas e categóricas
4. Criação de pipeline de pré-processamento:
   - imputação (SimpleImputer)
   - normalização (StandardScaler)
   - encoding (OneHotEncoder)
5. Divisão treino/teste com estratificação
6. Treinamento dos modelos baseline:
   - DummyClassifier
   - Logistic Regression
7. Avaliação com múltiplas métricas

---

### Decisões técnicas

- Uso de `class_weight='balanced'` devido ao desbalanceamento
- Uso de `StratifiedKFold` para validação
- Uso de pipeline para evitar data leakage
- Uso de Recall e F1-score como métricas principais

---

### Problemas encontrados

- Presença de valores inválidos (`inf` e `NaN`) após feature engineering
- Dependência inicial de `df_encoded`
- Necessidade de separar melhor EDA e modelagem

---

### Aprendizados

- Baseline deve ser simples e independente de feature engineering avançado
- Pipeline é essencial para reprodutibilidade
- Métricas devem ser escolhidas com base no problema de negócio