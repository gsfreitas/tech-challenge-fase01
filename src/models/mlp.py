"""
Arquitetura MLP para classificação de churn.
"""

from __future__ import annotations

import torch
from torch import nn


class ChurnMLP(nn.Module):
    """
    MLP para predição de churn.

    Args:
        n_features: features da entrada
        hidden_dims: tupla com tamanhos das camadas escondidas
        dropout_rates: tupla com dropout rate de cada camada escondida
                      (deve ter mesmo tamanho que hidden_dims)
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

        # Monta as camadas
        layers: list[nn.Module] = []
        input_dim = n_features
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # saída: 1 neurônio
        layers.append(nn.Linear(input_dim, 1))

        # nn.Sequential aplica as camadas em ordem no forward
        self.network = nn.Sequential(*layers)

        # guarda os params
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Retorna logits
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna probabilidade de churn
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """
        Conta parâmetros treináveis
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)