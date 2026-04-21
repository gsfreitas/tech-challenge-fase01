══════════════════════════════════════════════════════════════════════
ANÁLISE DE CUSTO — Trade-off FP vs FN
══════════════════════════════════════════════════════════════════════
Premissa de negócio:
  Custo de 1 Falso Negativo (perder churner):  R$ 1,500.00
  Custo de 1 Falso Positivo (abordagem vã):    R$ 50.00
  Razão FN/FP: 30.0x

Threshold padrão (0.5):
  FP=210, FN=60, Recall=78.57%
  Custo total:  R$ 100,500.00

Threshold ótimo (mínimo custo):
  Threshold = 0.11
  FP=531, FN=0, Recall=100.00%
  Custo total:  R$ 26,550.00

Economia ao usar threshold ótimo: R$ 73,950.00 (73.6%)
══════════════════════════════════════════════════════════════════════