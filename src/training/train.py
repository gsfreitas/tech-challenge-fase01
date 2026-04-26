"""
Pipeline de treinamento dos modelos baseline.

Responsabilidades:
- Carregar e limpar dados (DataLoader + DataCleaner)
- Aplicar feature engineering (FeatureEngineer)
- Construir pipeline sklearn (preprocessor + modelo)
- Executar CV estratificada (5 folds) com múltiplas métricas
- Treinar em train set e avaliar em holdout
- Rastrear tudo no MLflow (params, métricas CV, métricas holdout, artefatos)
- Salvar pipelines treinados em src/models/*.pkl

Execução:
    python -m src.training.train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.preprocessing import (
    build_preprocessor,
    drop_id_columns,
    split_feature_types,
)
from src.utils.config import (
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    get_dataset_path,
    get_mlflow_tracking_uri,
    get_models_dir,
)
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Definição dos modelos
# ─────────────────────────────────────────────────────────────

def get_baseline_models() -> dict[str, object]:
    """Retorna dicionário {nome: estimador} dos modelos baseline."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            penalty="l2",
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=8,
        ),
    }


# ─────────────────────────────────────────────────────────────
# Carregamento e preparação dos dados
# ─────────────────────────────────────────────────────────────

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega dados brutos, limpa e aplica feature engineering.

    IMPORTANTE: feature engineering aqui produz apenas features
    determinísticas por linha (combinações, bins, flags). A codificação
    (one-hot, scaling) acontece dentro do Pipeline sklearn.
    """
    logger.info("Carregando dataset de %s", get_dataset_path())
    loader = DataLoader(file_path=get_dataset_path())
    df_raw = loader.load_data()

    logger.info("Limpando dados (sem cálculo de estatísticas globais)...")
    cleaner = DataCleaner(df=df_raw)
    df_clean = cleaner.clean_data()

    logger.info("Aplicando feature engineering...")
    fe = FeatureEngineer(df=df_clean)
    df_fe = fe.create_family_status()
    df_fe = fe.create_tenure_bins()
    df_fe = fe.create_family_size_proxy()
    df_fe = fe.create_risk_profile()
    df_fe = fe.create_risk_combo()
    df_fe = fe.create_support_bundle()
    df_fe = fe.create_diff_monthly_charges()

    # Target binário
    if df_fe[TARGET_COLUMN].dtype == object:
        df_fe[TARGET_COLUMN] = df_fe[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # Dropar IDs
    df_fe = drop_id_columns(df_fe)

    # TenureBin é Categorical, converte para string para uniformizar
    if "TenureBin" in df_fe.columns:
        df_fe["TenureBin"] = df_fe["TenureBin"].astype(str)

    y = df_fe[TARGET_COLUMN]
    X = df_fe.drop(columns=[TARGET_COLUMN])

    logger.info("Dataset preparado: X=%s, y=%s (churn rate=%.2f%%)",
                X.shape, y.shape, 100 * y.mean())
    return X, y


# ─────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────

SCORING = {
    "accuracy": "accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def compute_holdout_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de holdout para um pipeline treinado."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "holdout_accuracy": accuracy_score(y_test, y_pred),
        "holdout_f1": f1_score(y_test, y_pred, zero_division=0),
        "holdout_precision": precision_score(y_test, y_pred, zero_division=0),
        "holdout_recall": recall_score(y_test, y_pred, zero_division=0),
    }

    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["holdout_roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["holdout_pr_auc"] = average_precision_score(y_test, y_proba)

    return metrics


def cv_summary(cv_scores: dict) -> dict[str, float]:
    """Extrai média e desvio padrão das métricas de CV."""
    summary = {}
    for metric_name in SCORING:
        key = f"test_{metric_name}"
        if key in cv_scores:
            summary[f"cv_{metric_name}_mean"] = float(cv_scores[key].mean())
            summary[f"cv_{metric_name}_std"] = float(cv_scores[key].std())
    return summary


# ─────────────────────────────────────────────────────────────
# Treino de um modelo individual (CV + holdout + MLflow)
# ─────────────────────────────────────────────────────────────

def train_single_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    models_dir: Path,
) -> dict[str, float]:
    """
    Treina um modelo: CV estratificada + fit final + avaliação holdout + MLflow.

    Retorna as métricas consolidadas (para tabela comparativa).
    """
    logger.info("═══ Treinando: %s ═══", model_name)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Executando CV estratificada (5 folds)...")
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=False
    )
    cv_metrics = cv_summary(cv_scores)

    logger.info("Fit em X_train completo...")
    pipeline.fit(X_train, y_train)

    logger.info("Avaliando em holdout...")
    holdout_metrics = compute_holdout_metrics(pipeline, X_test, y_test)

    all_metrics = {**cv_metrics, **holdout_metrics}

    # MLflow tracking
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_features_numeric", len(numeric_features))
        mlflow.log_param("n_features_categorical", len(categorical_features))

        # params do modelo
        for k, v in model.get_params().items():
            try:
                mlflow.log_param(f"model__{k}", v)
            except Exception:
                pass  # ignora params não-serializáveis

        mlflow.log_metrics(all_metrics)

        # Log do pipeline como modelo sklearn
        mlflow.sklearn.log_model(pipeline, name="model")

        # Salva .pkl local + registra como artefato
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        # Registra no Model Registry como Production
        # Necessário pra API carregar via `models:/<n>/Production`.
        # Pula o dummy (não faz sentido servir baseline trivial).
        if model_name != "dummy":
            try:
                model_uri = f"runs:/{run.info.run_id}/model"
                registered = mlflow.register_model(model_uri, name=model_name)

                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "Modelo '%s' registrado no Registry como v%s [Production]",
                    model_name, registered.version,
                )
            except Exception as e:
                logger.warning("Falha ao registrar %s no Registry: %s", model_name, e)

        logger.info(
            "%s -> ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Recall=%.4f",
            model_name,
            all_metrics.get("holdout_roc_auc", np.nan),
            all_metrics.get("holdout_pr_auc", np.nan),
            all_metrics.get("holdout_f1", np.nan),
            all_metrics.get("holdout_recall", np.nan),
        )

    return {"model": model_name, **all_metrics}


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def run_training(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """Orquestra o treino de todos os baselines e retorna tabela comparativa."""
    set_global_seed(RANDOM_STATE)

    # MLflow: tracking na raiz do projeto
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("MLflow experiment: %s", experiment_name)

    # Dados
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split: train=%s, test=%s (churn rate train=%.2f%%, test=%.2f%%)",
        X_train.shape, X_test.shape, 100 * y_train.mean(), 100 * y_test.mean(),
    )

    numeric_features, categorical_features = split_feature_types(X_train)
    models_dir = get_models_dir()

    # Treino dos baselines
    results = []
    for name, model in get_baseline_models().items():
        result = train_single_model(
            model_name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            models_dir=models_dir,
        )
        results.append(result)

    # Tabela comparativa
    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n═══ Comparativo final (holdout) ═══\n%s",
                results_df[[c for c in results_df.columns if c.startswith("holdout_")]])

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos baseline de churn.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help="Nome do experimento MLflow.",
    )
    args = parser.parse_args()

    setup_logging()
    results_df = run_training(experiment_name=args.experiment)

    logger.info("Treino concluído. Runs registrados em %s", get_mlflow_tracking_uri())
    print("\n", results_df.round(4).to_string(), "\n")


if __name__ == "__main__":
    main()