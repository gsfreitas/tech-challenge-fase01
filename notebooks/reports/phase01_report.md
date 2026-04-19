# Tech Challenge — Fase 01  
## Predição de Churn de Clientes

---

## 1. Objetivo

O objetivo desta fase é compreender o comportamento dos dados, identificar fatores associados ao churn de clientes e construir um modelo baseline para validação inicial do problema.

---

## 2. Descrição dos Dados

- Dataset: IBM Telco Customer Churn  
- Número de registros: 7043  
- Variável alvo: `Churn` (convertida para variável binária `target`)

### Tipos de variáveis

- Demográficas: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Contratuais: `Contract`, `PaymentMethod`, `PaperlessBilling`
- Serviços: `InternetService`, `OnlineSecurity`, `TechSupport`, entre outros
- Numéricas: `tenure`, `MonthlyCharges`, `TotalCharges`

---

## 3. Análise Exploratória de Dados (EDA)

### 3.1 Qualidade dos Dados

- Identificação de valores ausentes na variável `TotalCharges`
- Conversão para tipo numérico
- Tratamento realizado com imputação pela mediana

### 3.2 Balanceamento da Variável Alvo

- Classe negativa (No Churn): aproximadamente 73%  
- Classe positiva (Churn): aproximadamente 27%  

Conclusão:
- Dataset desbalanceado, exigindo atenção na escolha das métricas e validação

---

### 3.3 Distribuição das Variáveis

- `tenure`: distribuição aproximadamente simétrica  
- `MonthlyCharges`: distribuição relativamente equilibrada  
- `TotalCharges`: assimetria à direita  
- `SeniorCitizen`: distribuição altamente concentrada  

Implicações:
- Presença de assimetrias em variáveis numéricas
- Necessidade de normalização na modelagem

---

### 3.4 Outliers

- Avaliação realizada com boxplots e análise estatística
- Não foram identificados outliers extremos relevantes

Conclusão:
- Não foi necessária remoção de registros

---

### 3.5 Correlação

- Avaliação por matriz de correlação
- Identificação de relações entre variáveis numéricas
- Possível redundância entre variáveis derivadas

---

## 4. Feature Analysis

### 4.1 Tipo de Contrato

- Clientes com contrato `Month-to-month` apresentam maior churn  
- Contratos de longo prazo apresentam menor churn  

Conclusão:
- Variável com forte poder explicativo

---

### 4.2 Método de Pagamento

- `Electronic check` apresenta maior taxa de churn  
- Métodos automáticos apresentam maior retenção  

Hipótese:
- Automação reduz churn involuntário
- Método de pagamento reflete comportamento do cliente

---

### 4.3 Serviços Adicionais

- Ausência de serviços adicionais está associada a maior churn  
- Presença de serviços como suporte e segurança reduz churn  

Conclusão:
- Valor percebido influencia retenção

---

### 4.4 Perfil Familiar

- Clientes sem parceiro e dependentes apresentam maior churn  
- Clientes com estrutura familiar apresentam maior retenção  

---

### 4.5 Variáveis Numéricas

- Baixo `tenure` está associado a maior churn  
- `MonthlyCharges` mais elevados indicam maior propensão ao churn  

---

## 5. Feature Engineering

Foram criadas variáveis derivadas para capturar padrões comportamentais:

- `RiskProfile`
- `FamilyStatus`
- `FamilySizeProxy`
- `RiskCombo`
- `SupportBundle`
- `ChargeDiff`

Essas variáveis visam representar interações entre serviços, perfil do cliente e comportamento de consumo.

---

## 6. Modelo Baseline

### 6.1 Abordagem

Foram utilizados dois modelos:

- DummyClassifier (baseline de referência)
- Regressão Logística

### 6.2 Pipeline de Modelagem

- Separação de features e target
- Identificação de variáveis numéricas e categóricas
- Pré-processamento com:
  - imputação
  - normalização
  - encoding
- Divisão treino/teste com estratificação
- Validação cruzada estratificada

---

### 6.3 Métricas Utilizadas

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Observação:
- Devido ao desbalanceamento, Recall e F1-score foram priorizados

---

### 6.4 Resultados — Regressão Logística

- Accuracy: 0.76  
- Precision: 0.53  
- Recall: 0.81  
- F1 Score: 0.64  

---

### 6.5 Interpretação

- O modelo apresenta alto recall, indicando boa capacidade de identificar clientes com churn  
- A precisão é menor, indicando presença de falsos positivos  
- Accuracy não é adequada como métrica principal neste contexto  

Conclusão:
- O modelo é adequado como baseline inicial
- Existe espaço para melhoria na precisão

---

### 6.6 Matriz de Confusão

- Boa identificação da classe positiva (churn)
- Erros concentrados em falsos positivos

Interpretação:
- Estratégia conservadora, priorizando detecção de churn

---

## 7. Desafios Identificados

- Dataset desbalanceado  
- Variáveis com assimetria  
- Possível redundância entre variáveis  
- Necessidade de pipeline consistente  
- Separação clara entre EDA, feature engineering e modelagem  

---

## 8. Próximos Passos

- Refinar feature engineering
- Testar modelos mais robustos
- Ajustar hiperparâmetros
- Trabalhar threshold de decisão
- Avaliar técnicas de balanceamento
- Implementar rastreamento de experimentos (MLflow)

---

## 9. Conclusão

A análise exploratória identificou padrões relevantes de churn, especialmente relacionados a contrato, método de pagamento e perfil do cliente.

O modelo baseline com regressão logística apresentou desempenho consistente, validando a viabilidade do problema e a qualidade dos dados.

A próxima fase focará na melhoria da performance e na robustez do modelo.