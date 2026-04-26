import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training.train_mlp import (
    split_train_val_test,
    train_one_epoch,
    evaluate,
    compute_metrics,
)


# ======================
# SPLIT
# ======================


class TestSplitTrainValTest:

    def test_split_shapes(self):
        """
        Given: dataset simples
        When: split é executado
        Then: deve dividir corretamente
        """
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0, 1] * 50)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_val) + len(X_test) == 100


# ======================
# TRAIN ONE EPOCH
# ======================


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class TestTrainOneEpoch:

    def test_returns_loss(self):
        """
        Given: modelo simples e loader fake
        When: treino roda
        Then: retorna loss float
        """
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")

        X = torch.randn(10, 2)
        y = torch.randint(0, 2, (10, 1)).float()

        loader = [(X, y)]

        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)

        assert isinstance(loss, float)


# ======================
# EVALUATE
# ======================


class TestEvaluate:

    def test_returns_correct_outputs(self):
        """
        Given: modelo simples
        When: evaluate roda
        Then: retorna loss, probs e targets
        """
        model = DummyModel()
        loss_fn = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")

        X = torch.randn(10, 2)
        y = torch.randint(0, 2, (10, 1)).float()

        loader = [(X, y)]

        loss, probs, targets = evaluate(model, loader, loss_fn, device)

        assert isinstance(loss, float)
        assert isinstance(probs, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert len(probs) == len(targets)


# ======================
# METRICS
# ======================


class TestComputeMetrics:

    def test_returns_all_metrics(self):
        """
        Given: dados simples
        When: compute_metrics roda
        Then: retorna todas métricas
        """
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = compute_metrics(y_true, y_proba)

        expected_keys = {
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "pr_auc",
        }

        assert set(metrics.keys()) == expected_keys

    def test_threshold_behavior(self):
        """
        Given: probabilidades
        When: threshold muda
        Then: predição muda
        """
        y_true = np.array([0, 1])
        y_proba = np.array([0.4, 0.6])

        metrics = compute_metrics(y_true, y_proba, threshold=0.5)

        assert metrics["accuracy"] == 1.0
