"""
Arquitetura MLP (Multi-Layer Perceptron) para classificação binária de churn.

Design:
    Input (n_features)
      -> Linear(128) + ReLU + Dropout(0.3)
      -> Linear(64)  + ReLU + Dropout(0.3)
      -> Linear(32)  + ReLU + Dropout(0.2)
      -> Linear(1)   (logits crus; sigmoid aplicada dentro da loss)
"""

from __future__ import annotations

import torch
from torch import nn


class ChurnMLP(nn.Module):
    """
    MLP para predição de churn.

    Args:
        n_features: dimensão da entrada (número de features após pré-processamento)
        hidden_dims: tupla com tamanhos das camadas escondidas
        dropout_rates: tupla com dropout rate de cada camada escondida
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout_rates: tuple[float, ...] = (0.3, 0.3, 0.2),
    ):
        super().__init__()

        if len(hidden_dims) != len(dropout_rates):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) e dropout_rates "
                f"({len(dropout_rates)}) devem ter mesmo tamanho."
            )

        # monta as camadas dinamicamente
        layers: list[nn.Module] = []
        input_dim = n_features
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # camada de saída: 1 neurônio
        layers.append(nn.Linear(input_dim, 1))

        # nn.Sequential aplica as camadas em ordem no forward
        self.network = nn.Sequential(*layers)

        # guarda os hiperparametros
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Retorna logits (não probabilidades).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna probabilidade de churn (entre 0 e 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """
        Conta parâmetros treináveis — útil para logar no MLflow
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)