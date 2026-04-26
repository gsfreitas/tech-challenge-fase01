# tests/unit/test_train.py

import contextlib

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.training.train import (
    compute_holdout_metrics,
    cv_summary,
    get_baseline_models,
    run_training,
    train_single_model,
)


class TestGetBaselineModels:
    """Testes dos modelos baseline."""

    def test_returns_expected_models(self):
        """
        Given: chamada da função
        When: get_baseline_models é executado
        Then: retorna os 3 modelos esperados
        """
        models = get_baseline_models()

        assert "dummy" in models
        assert "logistic_regression" in models
        assert "decision_tree" in models
        assert len(models) == 3


class TestComputeHoldoutMetrics:
    """Testes das métricas de holdout."""

    def test_returns_basic_metrics(self):
        """
        Given: pipeline simples treinado
        When: compute_holdout_metrics é chamado
        Then: retorna métricas básicas
        """
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])

        pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, y)

        metrics = compute_holdout_metrics(pipeline, X, y)

        assert "holdout_accuracy" in metrics
        assert "holdout_f1" in metrics
        assert "holdout_precision" in metrics
        assert "holdout_recall" in metrics

    def test_includes_probability_metrics_when_available(self):
        """
        Given: modelo com predict_proba
        When: compute_holdout_metrics é chamado
        Then: inclui ROC-AUC e PR-AUC
        """
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])

        pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, y)

        metrics = compute_holdout_metrics(pipeline, X, y)

        assert "holdout_roc_auc" in metrics
        assert "holdout_pr_auc" in metrics

    def test_metrics_values_are_between_zero_and_one(self):
        """
        Given: pipeline simples treinado
        When: compute_holdout_metrics é chamado
        Then: todas as métricas devem estar entre 0 e 1
        """
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])

        pipeline = Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
        pipeline.fit(X, y)

        metrics = compute_holdout_metrics(pipeline, X, y)

        for value in metrics.values():
            assert 0 <= value <= 1


class TestCvSummary:
    """Testes do resumo das métricas de cross-validation."""

    def test_computes_mean_and_std(self):
        """
        Given: dicionário simulando saída do cross_validate
        When: cv_summary é chamado
        Then: calcula média e desvio padrão das métricas existentes
        """
        scores = {
            "test_accuracy": np.array([0.8, 0.9, 1.0]),
            "test_f1": np.array([0.7, 0.8, 0.9]),
        }

        summary = cv_summary(scores)

        assert "cv_accuracy_mean" in summary
        assert "cv_accuracy_std" in summary
        assert "cv_f1_mean" in summary
        assert "cv_f1_std" in summary

    def test_ignores_missing_metrics(self):
        """
        Given: dicionário com apenas accuracy
        When: cv_summary é chamado
        Then: retorna somente métricas disponíveis
        """
        scores = {
            "test_accuracy": np.array([0.5, 0.5, 0.5]),
        }

        summary = cv_summary(scores)

        assert summary["cv_accuracy_mean"] == 0.5
        assert "cv_f1_mean" not in summary


class TestLoadAndPrepareData:
    """Testes de load_and_prepare_data usando mocks."""

    def test_returns_X_y(self, monkeypatch, sample_data):
        """
        Given: mocks de loader, cleaner e feature engineer usando sample_data
        When: load_and_prepare_data é chamado
        Then: retorna X sem target/customerID e y como Series binária
        """
        df = sample_data.copy()

        monkeypatch.setattr(
            "src.training.train.DataLoader",
            lambda file_path: type(
                "MockLoader",
                (),
                {"load_data": lambda self: df},
            )(),
        )

        monkeypatch.setattr(
            "src.training.train.DataCleaner",
            lambda df: type(
                "MockCleaner",
                (),
                {"clean_data": lambda self: df},
            )(),
        )

        monkeypatch.setattr(
            "src.training.train.FeatureEngineer",
            lambda df: type(
                "MockFeatureEngineer",
                (),
                {
                    "create_family_status": lambda self: df,
                    "create_tenure_bins": lambda self: df,
                    "create_family_size_proxy": lambda self: df,
                    "create_risk_profile": lambda self: df,
                    "create_risk_combo": lambda self: df,
                    "create_support_bundle": lambda self: df,
                    "create_diff_monthly_charges": lambda self: df,
                },
            )(),
        )

        from src.training.train import load_and_prepare_data

        X, y = load_and_prepare_data()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "Churn" not in X.columns
        assert "customerID" not in X.columns
        assert set(y.unique()).issubset({0, 1})


class TestTrainSingleModel:
    """Testes de train_single_model usando mocks para MLflow e I/O."""

    def test_runs_without_error(self, monkeypatch, tmp_path):
        """
        Given: dados simples e mocks das dependências externas
        When: train_single_model é chamado
        Then: retorna dicionário com nome do modelo e métricas
        """
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)

        fake_cv_scores = {
            "test_accuracy": np.array([0.8, 0.9]),
            "test_f1": np.array([0.7, 0.8]),
            "test_precision": np.array([0.7, 0.8]),
            "test_recall": np.array([0.7, 0.8]),
            "test_roc_auc": np.array([0.8, 0.9]),
            "test_pr_auc": np.array([0.8, 0.9]),
        }

        monkeypatch.setattr(
            "src.training.train.cross_validate",
            lambda *args, **kwargs: fake_cv_scores,
        )

        monkeypatch.setattr(
            "src.training.train.mlflow.start_run",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            "src.training.train.mlflow.log_param",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.train.mlflow.log_metrics",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.train.mlflow.sklearn.log_model",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.train.mlflow.log_artifact",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.train.joblib.dump",
            lambda *args, **kwargs: None,
        )

        result = train_single_model(
            model_name="dummy",
            model=DummyClassifier(strategy="most_frequent"),
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            numeric_features=["a"],
            categorical_features=[],
            models_dir=tmp_path,
        )

        assert result["model"] == "dummy"
        assert "cv_accuracy_mean" in result
        assert "holdout_accuracy" in result


class TestRunTraining:
    """Testes de run_training usando mocks para evitar treino real."""

    def test_runs_without_error(self, monkeypatch, tmp_path):
        """
        Given: mocks das funções pesadas do treinamento
        When: run_training é chamado
        Then: retorna DataFrame com resultado dos modelos
        """
        X = pd.DataFrame({"feature": list(range(20))})
        y = pd.Series([0, 1] * 10)

        monkeypatch.setattr(
            "src.training.train.load_and_prepare_data",
            lambda: (X, y),
        )

        monkeypatch.setattr(
            "src.training.train.split_feature_types",
            lambda X_train: (["feature"], []),
        )

        monkeypatch.setattr(
            "src.training.train.get_models_dir",
            lambda *args, **kwargs: tmp_path,
        )

        monkeypatch.setattr(
            "src.training.train.get_baseline_models",
            lambda: {"dummy": object()},
        )

        monkeypatch.setattr(
            "src.training.train.train_single_model",
            lambda **kwargs: {"model": "dummy", "holdout_accuracy": 1.0},
        )

        monkeypatch.setattr(
            "src.training.train.mlflow.set_tracking_uri",
            lambda *args, **kwargs: None,
        )

        monkeypatch.setattr(
            "src.training.train.mlflow.set_experiment",
            lambda *args, **kwargs: None,
        )

        monkeypatch.setattr(
            "src.training.train.mlflow.get_tracking_uri",
            lambda *args, **kwargs: "mock",
        )

        result = run_training()

        assert isinstance(result, pd.DataFrame)
        assert "holdout_accuracy" in result.columns
        assert "dummy" in result.index
