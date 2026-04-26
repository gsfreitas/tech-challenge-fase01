"""
Análise de trade-off Custo FP vs FN — encontra o threshold ótimo de negócio
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.features.preprocessing import build_preprocessor, split_feature_types
from src.models.mlp import ChurnMLP
from src.training.train import load_and_prepare_data
from src.training.train_mlp import split_train_val_test
from src.utils.config import (
    DOCS_DIR,
    MLP_DROPOUT_RATES,
    MLP_HIDDEN_DIMS,
    MLP_VAL_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    get_mlflow_tracking_uri,
    get_models_dir,
)
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)

DEFAULT_COST_FN = 1500.0
DEFAULT_COST_FP = 50.0


def compute_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """custo total"""
    y_pred = (y_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = fn * cost_fn + fp * cost_fp
    n_churners = int(tp + fn)
    recall = tp / n_churners if n_churners > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_flagged": int(tp + fp),
        "recall": recall,
        "precision": precision,
        "cost_fn_total": fn * cost_fn,
        "cost_fp_total": fp * cost_fp,
        "total_cost": total_cost,
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Varre thresholds e retorna df com os custos
    """
    thresholds = np.arange(0.05, 0.96, step)
    rows = [compute_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds]
    return pd.DataFrame(rows)


def load_best_model_and_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduz o split do train_mlp.py, carrega o MLP salvo, gera probas no test set.
    Usa o MLP porque é o modelo central do tech challenge
    """
    set_global_seed(RANDOM_STATE)

    # reproduz o split do train_mlp.py
    X, y = load_and_prepare_data()
    _, _, X_test, _, _, y_test = split_train_val_test(
        X, y,
        test_size=TEST_SIZE * 0.75,
        val_size=MLP_VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    # carrega preprocessor fitado
    preprocessor = joblib.load(get_models_dir() / "mlp_preprocessor.pkl")
    X_test_prep = preprocessor.transform(X_test)

    # recria arquitetura e carrega pesos
    n_features = X_test_prep.shape[1]
    model = ChurnMLP(
        n_features=n_features,
        hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rates=MLP_DROPOUT_RATES,
    )
    model.load_state_dict(torch.load(get_models_dir() / "mlp.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test_prep.astype(np.float32))
        logits = model(X_tensor)
        probas = torch.sigmoid(logits).numpy().ravel()

    return y_test.values, probas


def format_summary(df: pd.DataFrame, cost_fn: float, cost_fp: float) -> str:
    """gera texto com resumo dos resultados."""
    optimal = df.loc[df["total_cost"].idxmin()]
    default = df.iloc[(df["threshold"] - 0.5).abs().argmin()]

    savings = default["total_cost"] - optimal["total_cost"]
    savings_pct = 100 * savings / default["total_cost"] if default["total_cost"] > 0 else 0

    lines = [
        "═" * 70,
        "ANÁLISE DE CUSTO — Trade-off FP vs FN",
        "═" * 70,
        f"Premissa de negócio:",
        f"  Custo de 1 Falso Negativo (perder churner):  R$ {cost_fn:,.2f}",
        f"  Custo de 1 Falso Positivo (abordagem vã):    R$ {cost_fp:,.2f}",
        f"  Razão FN/FP: {cost_fn / cost_fp:.1f}x",
        "",
        f"Threshold padrão (0.5):",
        f"  FP={int(default['fp'])}, FN={int(default['fn'])}, Recall={default['recall']:.2%}",
        f"  Custo total:  R$ {default['total_cost']:,.2f}",
        "",
        f"Threshold ótimo (mínimo custo):",
        f"  Threshold = {optimal['threshold']:.2f}",
        f"  FP={int(optimal['fp'])}, FN={int(optimal['fn'])}, Recall={optimal['recall']:.2%}",
        f"  Custo total:  R$ {optimal['total_cost']:,.2f}",
        "",
        f"Economia ao usar threshold ótimo: R$ {savings:,.2f} ({savings_pct:.1f}%)",
        "═" * 70,
    ]
    return "\n".join(lines)


def main() -> None: # pragma: no cover
    parser = argparse.ArgumentParser(description="Análise de custo FP vs FN.")
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN,
                        help="Custo de 1 Falso Negativo em R$")
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP,
                        help="Custo de 1 Falso Positivo em R$")
    args = parser.parse_args()

    setup_logging()

    logger.info("Carregando modelo e dados...")
    y_test, probas = load_best_model_and_data()
    logger.info("Test set: %d amostras (churn rate=%.2f%%)",
                len(y_test), 100 * y_test.mean())

    logger.info("Varrendo thresholds de 0.05 a 0.95...")
    df = sweep_thresholds(y_test, probas, args.cost_fn, args.cost_fp)

    # Salva CSV
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DOCS_DIR / "cost_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info("CSV salvo em %s", csv_path)

    summary = format_summary(df, args.cost_fn, args.cost_fp)
    print("\n" + summary + "\n")

    # salva sumário
    summary_path = DOCS_DIR / "cost_analysis_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Sumário salvo em %s", summary_path)

    # log do threshold ótimo
    optimal = df.loc[df["total_cost"].idxmin()]
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    logger.info("Threshold ótimo identificado: %.2f", optimal["threshold"])


if __name__ == "__main__":
    main()