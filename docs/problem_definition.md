# Definição do Problema

## Contexto de negócio
Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes. O objetivo é identificar clientes com maior probabilidade de churn para apoiar ações de retenção.

## Problema de machine learning
Construir um modelo de classificação binária capaz de prever se um cliente irá cancelar o serviço.

## Variável alvo
- `Churn`

## Objetivo do projeto
Apoiar a tomada de decisão da área de retenção, priorizando clientes com maior risco de evasão.

## Stakeholders
- Diretoria
- Time de retenção
- Time de CRM/Marketing
- Time de dados

## Hipótese inicial
Características como tipo de contrato, tempo de permanência, cobrança mensal, método de pagamento e serviços adicionais podem influenciar o risco de churn.

## Métrica técnica principal
- ROC-AUC

## Métricas técnicas secundárias
- PR-AUC
- F1-score
- Recall
- Precision

## Métrica de negócio
- custo de churn evitado
- quantidade de clientes corretamente priorizados para retenção

## Riscos
- falsos negativos: perder clientes com alto risco real
- falsos positivos: gastar esforço com clientes que não cancelariam
- desbalanceamento da classe alvo
- vazamento de dados no pré-processamento

## Restrições
- dataset tabular
- classificação binária
- necessidade de comparação com baseline
- necessidade de rastreamento com MLflow