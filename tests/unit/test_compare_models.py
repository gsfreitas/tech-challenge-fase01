# tests/unit/test_compare_models.py

import pandas as pd
import pytest

from src.training.compare_models import (
    METRIC_ORDER,
    build_comparison_table,
    format_markdown_table,
    identify_best_model,
    load_runs_from_mlflow,
    save_artifacts,
)


def make_runs_df():
    data = {
        "tags.mlflow.runName": ["dummy", "dummy", "logistic_regression"],
        "start_time": [1, 2, 3],
    }

    for metric in METRIC_ORDER:
        data[f"metrics.{metric}"] = [0.5, 0.6, 0.8]

    return pd.DataFrame(data)


class TestLoadRunsFromMlflow:
    def test_load_runs_returns_dataframe(self, monkeypatch):
        """
        Given: experimento existente no MLflow
        When: load_runs_from_mlflow é chamado
        Then: retorna DataFrame de runs
        """
        runs = make_runs_df()

        class MockExperiment:
            experiment_id = "123"

        monkeypatch.setattr(
            "src.training.compare_models.mlflow.set_tracking_uri",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.compare_models.mlflow.get_experiment_by_name",
            lambda experiment_name: MockExperiment(),
        )
        monkeypatch.setattr(
            "src.training.compare_models.mlflow.search_runs",
            lambda *args, **kwargs: runs,
        )

        result = load_runs_from_mlflow("fake-experiment")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_load_runs_raises_when_experiment_not_found(self, monkeypatch):
        """
        Given: experimento inexistente
        When: load_runs_from_mlflow é chamado
        Then: levanta ValueError
        """
        monkeypatch.setattr(
            "src.training.compare_models.mlflow.set_tracking_uri",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.training.compare_models.mlflow.get_experiment_by_name",
            lambda experiment_name: None,
        )

        with pytest.raises(ValueError):
            load_runs_from_mlflow("missing")


class TestBuildComparisonTable:
    def test_builds_table_with_latest_runs(self):
        """
        Given: múltiplos runs por modelo
        When: build_comparison_table é chamado
        Then: mantém apenas o run mais recente de cada modelo
        """
        runs = make_runs_df()

        table = build_comparison_table(runs)

        assert "dummy" in table.index
        assert table.loc["dummy", "holdout_roc_auc"] == 0.6

    def test_table_has_correct_columns(self):
        """
        Given: runs válidos
        When: tabela é construída
        Then: colunas seguem METRIC_ORDER
        """
        runs = make_runs_df()

        table = build_comparison_table(runs)

        assert list(table.columns) == METRIC_ORDER


class TestFormatMarkdownTable:
    def test_returns_string(self):
        """
        Given: tabela comparativa
        When: format_markdown_table é chamado
        Then: retorna markdown em texto
        """
        table = build_comparison_table(make_runs_df())

        result = format_markdown_table(table)

        assert isinstance(result, str)
        assert "| Modelo |" in result
        assert "Dummy (baseline)" in result

    def test_formats_values_with_four_decimals(self):
        """
        Given: valores float
        When: markdown é gerado
        Then: valores aparecem com 4 casas decimais
        """
        table = build_comparison_table(make_runs_df())

        result = format_markdown_table(table)

        assert "0.6000" in result


class TestIdentifyBestModel:
    def test_returns_best_model_by_default_metric(self):
        """
        Given: tabela com modelos
        When: identify_best_model é chamado
        Then: retorna modelo com maior ROC-AUC
        """
        table = build_comparison_table(make_runs_df())

        model, score = identify_best_model(table)

        assert model == "logistic_regression"
        assert score == 0.8


class TestSaveArtifacts:
    def test_creates_csv_and_markdown_files(self, tmp_path, monkeypatch):
        """
        Given: tabela e markdown
        When: save_artifacts é chamado
        Then: cria arquivos em DOCS_DIR
        """
        table = build_comparison_table(make_runs_df())
        markdown = format_markdown_table(table)

        monkeypatch.setattr("src.training.compare_models.DOCS_DIR", tmp_path)

        save_artifacts(table, markdown)

        assert (tmp_path / "model_comparison.csv").exists()
        assert (tmp_path / "model_comparison.md").exists()
