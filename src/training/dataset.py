"""
Datasets e DataLoaders PyTorch para o problema de churn.
A ponte entre o mundo pandas (DataFrame) e PyTorch (Tensor)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ChurnDataset(Dataset):
    """
    Dataset PyTorch para features tabulares de churn

    Args:
        X: array 2D (n_samples, n_features)
        y: array 1D (n_samples,) com targets binários
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # converte para float32, dado que pythorch trabalha com essa arquitetura como padrão
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constrói os três DataLoaders (train/val/test).

    Decisões de design:
    - shuffle=True só no treino
    - num_workers=0 (default): carrega no processo principal
    - drop_last=False: não descarta a última batch
    """
    train_ds = ChurnDataset(X_train, y_train)
    val_ds = ChurnDataset(X_val, y_val)
    test_ds = ChurnDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, # só valida, não treina, logo não precisa embaralhar
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(
        "DataLoaders prontos: train=%s, val=%s, test=%s (batch_size=%s)",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        batch_size,
    )
    return train_loader, val_loader, test_loader


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """
    Calcula o equivalente do class_weight='balanced' do sklearn

    formula: n_negativos / n_positivos
    neste caso: ~5174 / ~1869 ≈ 2.77

    Isso diz à loss function: "cada erro em um churner custa 2.77× mais
    que um erro em um cliente retido".
    """
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    if n_pos == 0:
        raise ValueError(
            "Não é possível calcular pos_weight sem amostras positivas "
            f"(n_pos=0, n_neg={n_neg})."
        )
    pos_weight = float(n_neg) / float(n_pos)
    logger.info(
        "pos_weight=%.4f (n_neg=%d, n_pos=%d)", pos_weight, n_neg, n_pos
    )
    return torch.tensor([pos_weight], dtype=torch.float32)