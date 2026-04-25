"""
Comparação formal de modelos usando runs do MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd

from src.utils.config import (
    DOCS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    get_mlflow_tracking_uri,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

METRIC_ORDER = [
    "holdout_roc_auc",
    "holdout_pr_auc",
    "holdout_f1",
    "holdout_recall",
    "holdout_precision",
    "holdout_accuracy",
]

METRIC_LABELS = {
    "holdout_roc_auc": "ROC-AUC",
    "holdout_pr_auc": "PR-AUC",
    "holdout_f1": "F1",
    "holdout_recall": "Recall",
    "holdout_precision": "Precisão",
    "holdout_accuracy": "Acurácia",
}

MODEL_LABELS = {
    "dummy": "Dummy (baseline)",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "mlp_pytorch": "MLP (PyTorch)",
}

def load_runs_from_mlflow(experiment_name: str) -> pd.DataFrame:
    """
    Carrega todos os runs do experimento e retorna DataFrame
    """
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado no MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    logger.info("Encontrados %d runs no experimento '%s'", len(runs), experiment_name)
    return runs


def build_comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a tabela comparativa, pegando o run mais recente de cada modelo.
    """
    
    # Filtra apenas runs com nome reconhecido
    runs = runs[runs["tags.mlflow.runName"].isin(MODEL_LABELS.keys())].copy()

    # Pega o mais recente
    runs = runs.sort_values("start_time", ascending=False)
    runs = runs.drop_duplicates(subset=["tags.mlflow.runName"], keep="first")

    # monta tabela
    table_data = {}
    for _, run in runs.iterrows():
        model_name = run["tags.mlflow.runName"]
        table_data[model_name] = {
            metric: run.get(f"metrics.{metric}")
            for metric in METRIC_ORDER
        }

    table = pd.DataFrame(table_data).T
    table = table[METRIC_ORDER]

    # reordena linhas
    ordered_models = [m for m in MODEL_LABELS if m in table.index]
    table = table.loc[ordered_models]

    return table


def format_markdown_table(table: pd.DataFrame) -> str:
    """
    Formata a tabela como Markdown (resultados no README)
    """
    lines = []
    header = "| Modelo | " + " | ".join(METRIC_LABELS[m] for m in METRIC_ORDER) + " |"
    separator = "|" + "---|" * (len(METRIC_ORDER) + 1)
    lines.append(header)
    lines.append(separator)

    for model_name in table.index:
        label = MODEL_LABELS[model_name]
        metric_cells = [f"{table.loc[model_name, m]:.4f}" for m in METRIC_ORDER]
        lines.append(f"| {label} | " + " | ".join(metric_cells) + " |")

    return "\n".join(lines)


def identify_best_model(table: pd.DataFrame, metric: str = "holdout_roc_auc") -> tuple[str, float]:
    """
    Identifica o modelo com a melhor métrica (ROC-AUC)
    """
    best_idx = table[metric].idxmax()
    best_score = table.loc[best_idx, metric]
    return best_idx, best_score


def save_artifacts(table: pd.DataFrame, markdown: str) -> None:
    """
    salva CSV e Markdown
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DOCS_DIR / "model_comparison.csv"
    table.to_csv(csv_path)
    logger.info("CSV salvo em %s", csv_path)

    md_path = DOCS_DIR / "model_comparison.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown salvo em %s", md_path)


def main() -> None:
    setup_logging()

    logger.info("═══ Carregando runs do MLflow ═══")
    runs = load_runs_from_mlflow(MLFLOW_EXPERIMENT_NAME)

    logger.info("═══ Construindo tabela comparativa ═══")
    table = build_comparison_table(runs)

    best_model, best_score = identify_best_model(table)

    markdown = format_markdown_table(table)
    save_artifacts(table, markdown)

    print("\n" + "═" * 70)
    print("COMPARATIVO DE MODELOS — Tech Challenge Fase 01")
    print("═" * 70)
    print(table.round(4).to_string())
    print("\n" + "─" * 70)
    print(f"Melhor modelo (ROC-AUC): {MODEL_LABELS[best_model]} → {best_score:.4f}")
    print("─" * 70)
    print("\nTabela Markdown (cole no README):\n")
    print(markdown)
    print()


if __name__ == "__main__":
    main()