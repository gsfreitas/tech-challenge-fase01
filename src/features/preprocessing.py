"""
Pipeline de pré-processamento sklearn reutilizável.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Colunas que são dropadas antes do pipeline (identificadores)
ID_COLUMNS = ["customerID"]


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separa colunas do DataFrame em numéricas e categóricas.

    Args:
        X: DataFrame de features (sem target, sem customerID)

    Returns:
        (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    logger.info(
        "Detectadas %s features numéricas e %s categóricas.",
        len(numeric_features),
        len(categorical_features),
    )
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer padrão do projeto.

    - Numéricas: imputação por mediana + StandardScaler
    - Categóricas: imputação por moda + OneHotEncoder (handle_unknown='ignore')
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas de identificação que não devem entrar no modelo."""
    cols_to_drop = [c for c in ID_COLUMNS if c in df.columns]
    if cols_to_drop:
        logger.info("Removendo colunas de identificação: %s", cols_to_drop)
        return df.drop(columns=cols_to_drop)
    return df