# tests/unit/test_cost_analysis.py

import numpy as np
import pandas as pd
import torch

from src.analysis.cost_analysis import (
    compute_cost,
    format_summary,
    load_best_model_and_data,
    sweep_thresholds,
)


class TestComputeCost:
    """Testes da função compute_cost."""

    def test_basic_cost_calculation(self):
        """
        Given: valores reais e probabilidades
        When: compute_cost é chamado
        Then: calcula matriz de confusão e custo corretamente
        """
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2])

        result = compute_cost(
            y_true=y_true,
            y_proba=y_proba,
            threshold=0.5,
            cost_fn=100.0,
            cost_fp=10.0,
        )

        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tp"] == 2
        assert result["total_cost"] == 0

    def test_cost_with_false_negative(self):
        """
        Given: um churner não identificado
        When: compute_cost é chamado
        Then: custo de falso negativo deve ser aplicado
        """
        y_true = np.array([1])
        y_proba = np.array([0.2])

        result = compute_cost(y_true, y_proba, 0.5, 1500.0, 50.0)

        assert result["fn"] == 1
        assert result["cost_fn_total"] == 1500.0
        assert result["total_cost"] == 1500.0

    def test_cost_with_false_positive(self):
        """
        Given: um não churner marcado como churn
        When: compute_cost é chamado
        Then: custo de falso positivo deve ser aplicado
        """
        y_true = np.array([0])
        y_proba = np.array([0.9])

        result = compute_cost(y_true, y_proba, 0.5, 1500.0, 50.0)

        assert result["fp"] == 1
        assert result["cost_fp_total"] == 50.0
        assert result["total_cost"] == 50.0

    def test_no_churners_returns_zero_recall(self):
        """
        Given: nenhum churner real
        When: compute_cost é chamado
        Then: recall deve ser zero
        """
        y_true = np.array([0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3])

        result = compute_cost(y_true, y_proba, 0.5, 1500.0, 50.0)

        assert result["recall"] == 0.0

    def test_no_predicted_positive_returns_zero_precision(self):
        """
        Given: nenhuma predição positiva
        When: compute_cost é chamado
        Then: precision deve ser zero
        """
        y_true = np.array([1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3])

        result = compute_cost(y_true, y_proba, 0.5, 1500.0, 50.0)

        assert result["precision"] == 0.0


class TestSweepThresholds:
    """Testes da função sweep_thresholds."""

    def test_returns_dataframe(self):
        """
        Given: arrays de y_true e y_proba
        When: sweep_thresholds é chamado
        Then: retorna DataFrame com custos por threshold
        """
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9])

        result = sweep_thresholds(y_true, y_proba, 1500.0, 50.0, step=0.1)

        assert isinstance(result, pd.DataFrame)
        assert "threshold" in result.columns
        assert "total_cost" in result.columns

    def test_generates_multiple_thresholds(self):
        """
        Given: step definido
        When: sweep_thresholds é chamado
        Then: gera múltiplas linhas
        """
        y_true = np.array([0, 1])
        y_proba = np.array([0.2, 0.8])

        result = sweep_thresholds(y_true, y_proba, 1500.0, 50.0, step=0.2)

        assert len(result) > 1


class TestFormatSummary:
    """Testes da função format_summary."""

    def test_returns_summary_string(self):
        """
        Given: DataFrame de custos
        When: format_summary é chamado
        Then: retorna texto de resumo
        """
        df = pd.DataFrame(
            {
                "threshold": [0.5, 0.6],
                "fp": [10, 5],
                "fn": [2, 1],
                "tp": [8, 9],
                "recall": [0.8, 0.9],
                "total_cost": [3500.0, 2000.0],
            }
        )

        result = format_summary(df, cost_fn=1500.0, cost_fp=50.0)

        assert isinstance(result, str)
        assert "ANÁLISE DE CUSTO" in result
        assert "Threshold ótimo" in result

    def test_summary_handles_zero_default_cost(self):
        """
        Given: custo default igual a zero
        When: format_summary é chamado
        Then: não deve ocorrer divisão por zero
        """
        df = pd.DataFrame(
            {
                "threshold": [0.5, 0.6],
                "fp": [0, 0],
                "fn": [0, 0],
                "tp": [1, 1],
                "recall": [1.0, 1.0],
                "total_cost": [0.0, 0.0],
            }
        )

        result = format_summary(df, cost_fn=1500.0, cost_fp=50.0)

        assert "Economia" in result


class TestLoadBestModelAndData:
    """Teste de load_best_model_and_data com mocks."""

    def test_loads_model_and_returns_arrays(self, monkeypatch, tmp_path):
        """
        Given: mocks de dados, preprocessor e modelo salvo
        When: load_best_model_and_data é chamado
        Then: retorna y_test e probabilidades
        """
        X = pd.DataFrame(
            {
                "feature": list(range(20)),
            }
        )
        y = pd.Series([0, 1] * 10)

        class MockPreprocessor:
            def transform(self, X_test):
                return np.array([[1.0, 2.0] for _ in range(len(X_test))])

        class MockModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def load_state_dict(self, state_dict):
                return None

            def eval(self):
                return None

            def forward(self, x):
                return torch.zeros((x.shape[0], 1))

            def __call__(self, x):
                return self.forward(x)

        monkeypatch.setattr(
            "src.analysis.cost_analysis.load_and_prepare_data",
            lambda: (X, y),
        )

        monkeypatch.setattr(
            "src.analysis.cost_analysis.joblib.load",
            lambda path: MockPreprocessor(),
        )

        monkeypatch.setattr(
            "src.analysis.cost_analysis.ChurnMLP",
            MockModel,
        )

        monkeypatch.setattr(
            "src.analysis.cost_analysis.torch.load",
            lambda *args, **kwargs: {},
        )

        monkeypatch.setattr(
            "src.analysis.cost_analysis.get_models_dir",
            lambda: tmp_path,
        )

        y_test, probas = load_best_model_and_data()

        assert isinstance(y_test, np.ndarray)
        assert isinstance(probas, np.ndarray)
        assert len(y_test) == len(probas)
