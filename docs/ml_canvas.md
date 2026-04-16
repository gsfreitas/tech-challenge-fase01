# 🧩 ML Canvas

Este documento descreve a definição do problema de Machine Learning, conectando objetivos de negócio, dados, modelagem e estratégia de monitoramento para predição de churn em telecomunicações.

---

## 💼 Proposta de Valor

Este projeto busca reduzir o cancelamento de clientes (churn) em uma operadora de telecomunicações, problema que impacta diretamente a receita e a sustentabilidade do negócio.

A solução permite identificar clientes com alto risco de churn, possibilitando ações proativas de retenção e otimização de campanhas.

O objetivo é reduzir a taxa de churn, aumentar a retenção e preservar receita, otimizando campanhas e recursos.

---

## 🎯 Tarefa de ML

O objetivo é prever a probabilidade de churn de clientes.

Trata-se de um problema de classificação binária, respondendo à pergunta:
**“Este cliente irá cancelar o serviço no próximo período?”**

---

## 🧠 Decisões

As predições serão utilizadas para:

* Oferecer descontos ou benefícios
* Realizar contato proativo
* Priorizar clientes de alto valor
* Direcionar campanhas de retenção

Impacto esperado: redução do churn e aumento da retenção de clientes.

---

## 👥 Stakeholders

* Time de Marketing
* Diretoria
* Equipe de Atendimento
* Time de Dados

---

## 📊 Simulação de Impacto

O impacto do modelo pode ser avaliado por meio de simulações, considerando:

* Redução da taxa de churn
* Aumento da retenção
* Estimativa de receita preservada
* Otimização de campanhas

---

## 🔵 Fontes de Dados

Serão utilizados dados públicos de telecomunicações:

* IBM Telco Customer Churn Dataset
* Iranian Churn Dataset

---

## 🔵 Coleta de Dados

* Dados públicos já estruturados
* Importação via arquivos CSV
* Pré-processamento para padronização entre datasets

---

## 🧬 Variáveis (Features)

As principais variáveis incluem:

* Tempo de contrato (tenure)
* Valor mensal e total
* Tipo de serviço e plano
* Uso de serviços adicionais
* Histórico de comportamento do cliente

---

## 🏗️ Construção do Modelo

Os modelos são atualizados de forma iterativa durante o desenvolvimento, sempre que houver mudanças em features, parâmetros ou algoritmos.

Como os dados são estáticos, não há retreino automático. Em um cenário real, o modelo seria re-treinado periodicamente (ex: mensalmente) ou quando houver queda de performance ou data drift.

---

## 🧪 Avaliação Offline

A avaliação será realizada com validação cruzada estratificada e divisão treino/teste.

Serão utilizadas métricas como:

* AUC-ROC
* Recall
* F1-score
* Precision

Além disso, será feita comparação com modelos baseline.

Também será analisado o impacto de falsos positivos e falsos negativos no contexto de negócio.

---

## ⚡ Realização de Predições

As predições serão realizadas periodicamente sobre a base de clientes ativos, com frequência definida pelo negócio (diária ou semanal).

O processamento ocorrerá fora do horário crítico para não impactar os sistemas.

Em um cenário futuro, o modelo poderá evoluir para predições em tempo real.

---

## 🟣 Avaliação Online e Monitoramento

Em produção, o modelo será avaliado por métricas de negócio (churn, retenção, ROI) e métricas técnicas (AUC, Recall, F1).

Serão utilizados testes A/B para validação de impacto real.

Também serão monitorados latência, disponibilidade e drift (data e concept), com retreino do modelo em caso de degradação.

O monitoramento contínuo garante alinhamento entre performance do modelo e objetivos de negócio.

---

## 📏 SLOs (Service Level Objectives)

* Latência de predição: < 200ms (tempo real)
* Tempo de processamento batch: até algumas horas
* Disponibilidade do sistema: > 99%
* Atualização do modelo: mensal ou sob demanda

---

## 🔁 Atualização do Modelo

Durante o desenvolvimento, o modelo é atualizado iterativamente.

Em um cenário real, o retreino deve ocorrer periodicamente ou quando forem detectadas mudanças nos dados (data drift) ou queda de desempenho.
