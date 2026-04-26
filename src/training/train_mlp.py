"""
Pipeline de treinamento do MLP PyTorch.

Responsabilidades:
- Carregar e preparar dados
- Split train/val/test estratificado (70/15/15)
- Fit ColumnTransformer no treino
- Treinar MLP com BCEWithLogitsLoss + pos_weight + Adam
- Loop de treino com early stopping e logging por época
- Avaliação final no holdout test set
- Rastreamento completo no MLflow (hyperparâmetros, curvas, modelo)
- Salvar .pt local para uso pela API

Execução:
    python -m src.training.train_mlp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn, optim

from src.features.preprocessing import build_preprocessor, split_feature_types
from src.models.early_stopping import EarlyStopping
from src.models.mlp import ChurnMLP
from src.training.dataset import build_dataloaders, compute_pos_weight
from src.training.train import load_and_prepare_data  # reuso
from src.utils.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLP_BATCH_SIZE,
    MLP_DROPOUT_RATES,
    MLP_EARLY_STOPPING_PATIENCE,
    MLP_HIDDEN_DIMS,
    MLP_LEARNING_RATE,
    MLP_MAX_EPOCHS,
    MLP_VAL_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    get_mlflow_tracking_uri,
    get_models_dir,
)
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Split train/val/test
# ─────────────────────────────────────────────────────────────

def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split estratificado em três partes: train / val / test.

    Estratégia: dois splits sequenciais.
    1º split: separa test do resto (85/15)
    2º split: do restante (85%), separa val para ter 15/85
              o que equivale a 15% do total.
    """
    # Primeiro split: separa test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Segundo split: separa val do restante
    # Ajuste: val_size é proporção do total, mas aplicamos em X_trainval
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative_size,
        stratify=y_trainval,
        random_state=random_state,
    )
    logger.info(
        "Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(X_train), 100 * len(X_train) / len(X),
        len(X_val),   100 * len(X_val)   / len(X),
        len(X_test),  100 * len(X_test)  / len(X),
    )
    logger.info(
        "Churn rate: train=%.2f%%, val=%.2f%%, test=%.2f%%",
        100 * y_train.mean(), 100 * y_val.mean(), 100 * y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ───────────────────────────
# Loops de treino e avaliação
# ───────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Executa uma epoch de treino. Retorna loss média da epoch
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Avalia o modelo em um loader
    Usado tanto para validação durante treino quanto para avaliação final
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_samples
    probs_arr = np.concatenate(all_probs).ravel()
    targets_arr = np.concatenate(all_targets).ravel()
    return avg_loss, probs_arr, targets_arr


# ────────
# Métricas
# ────────

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calcula as 6 métricas do projeto
    """
    y_pred = (y_proba > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# ──────────────────
# Pipeline principal
# ──────────────────

def run_training(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> dict: # pragma: no cover
    """
    Orquestra o treino do MLP
    """
    set_global_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)

    # Carrega e prepara dados
    X, y = load_and_prepare_data()

    # Divide train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        test_size=TEST_SIZE * 0.75,  # 0.15
        val_size=MLP_VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    # fita o preprocessor no treino
    numeric_features, categorical_features = split_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    logger.info("Fitando preprocessor apenas no train set...")
    preprocessor.fit(X_train)

    X_train_prep = preprocessor.transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    n_features = X_train_prep.shape[1]
    logger.info("Features após preprocessing: %d", n_features)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train_prep, y_train.values,
        X_val_prep, y_val.values,
        X_test_prep, y_test.values,
        batch_size=MLP_BATCH_SIZE,
    )

    # modelo, loss, optimizer
    model = ChurnMLP(
        n_features=n_features,
        hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rates=MLP_DROPOUT_RATES,
    ).to(device)
    logger.info("Arquitetura:\n%s", model)
    logger.info("Parâmetros treináveis: %d", model.count_parameters())

    pos_weight = compute_pos_weight(y_train.values).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    early_stopping = EarlyStopping(
        patience=MLP_EARLY_STOPPING_PATIENCE, mode="min"
    )

    # MLflow
    with mlflow.start_run(run_name="mlp_pytorch") as run:
        # log hyperparameters
        mlflow.log_params({
            "model_type": "mlp_pytorch",
            "n_features": n_features,
            "hidden_dims": str(MLP_HIDDEN_DIMS),
            "dropout_rates": str(MLP_DROPOUT_RATES),
            "learning_rate": MLP_LEARNING_RATE,
            "batch_size": MLP_BATCH_SIZE,
            "max_epochs": MLP_MAX_EPOCHS,
            "early_stopping_patience": MLP_EARLY_STOPPING_PATIENCE,
            "pos_weight": float(pos_weight.item()),
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "random_state": RANDOM_STATE,
            "n_trainable_params": model.count_parameters(),
        })

        # loop de treino
        logger.info("═══ Iniciando treino ═══")
        for epoch in range(1, MLP_MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_probs, val_targets = evaluate(model, val_loader, loss_fn, device)
            val_metrics = compute_metrics(val_targets, val_probs)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_roc_auc=%.4f | val_recall=%.4f",
                epoch, train_loss, val_loss, val_metrics["roc_auc"], val_metrics["recall"],
            )

            early_stopping(val_loss, model)
            if early_stopping.should_stop:
                break

        # restaura melhor modelo e avalia no test
        early_stopping.restore_best(model)

        logger.info("═══ Avaliação final no test set ═══")
        _, test_probs, test_targets = evaluate(model, test_loader, loss_fn, device)
        test_metrics = compute_metrics(test_targets, test_probs)

        # log holdout_ metrics
        mlflow.log_metrics({f"holdout_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("best_epoch", early_stopping.best_epoch)

        logger.info(
            "MLP holdout: ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f",
            test_metrics["roc_auc"], test_metrics["pr_auc"],
            test_metrics["f1"], test_metrics["recall"], test_metrics["precision"],
        )

        # salva artefatos: modelo PyTorch + preprocessor
        models_dir = get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        # state_dict
        model_path = models_dir / "mlp.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        # preprocessor fitado
        preprocessor_path = models_dir / "mlp_preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(str(preprocessor_path))

        # MLflow pytorch log
        mlflow.pytorch.log_model(model, name="model")

        # Registra no Model Registry como Production
        # Necessário pra API carregar via `models:/mlp_pytorch/Production`.
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            registered = mlflow.register_model(model_uri, name="mlp_pytorch")

            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name="mlp_pytorch",
                version=registered.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                "Modelo 'mlp_pytorch' registrado no Registry como v%s [Production]",
                registered.version,
            )
        except Exception as e:
            logger.warning("Falha ao registrar mlp_pytorch no Registry: %s", e)

        logger.info("Artefatos salvos em %s", models_dir)

    return test_metrics


def main() -> None: # pragma: no cover
    parser = argparse.ArgumentParser(description="Treina MLP PyTorch para churn.")
    parser.add_argument(
        "--experiment", type=str, default=MLFLOW_EXPERIMENT_NAME,
        help="Nome do experimento MLflow.",
    )
    args = parser.parse_args()

    setup_logging()
    metrics = run_training(experiment_name=args.experiment)

    logger.info("Treino MLP concluído.")
    print("\n=== Métricas finais no test set ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()