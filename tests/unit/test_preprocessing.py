import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from src.features.preprocessing import (
    split_feature_types,
    build_preprocessor,
    drop_id_columns,
)


# ======================
# SPLIT FEATURE TYPES
# ======================


class TestSplitFeatureTypes:

    def test_splits_numeric_and_categorical(self):
        """
        Given: dataframe com tipos mistos
        When: split_feature_types é chamado
        Then: separa corretamente numéricas e categóricas
        """
        df = pd.DataFrame(
            {
                "age": [20, 30],
                "income": [1000.0, 2000.0],
                "gender": ["M", "F"],
                "is_active": [True, False],
            }
        )

        num, cat = split_feature_types(df)

        assert "age" in num
        assert "income" in num
        assert "gender" in cat
        assert "is_active" in cat

    def test_returns_empty_when_no_type(self):
        """
        Given: dataframe só com numéricas
        When: split_feature_types é chamado
        Then: categóricas deve ser vazio
        """
        df = pd.DataFrame(
            {
                "age": [20, 30],
                "income": [1000, 2000],
            }
        )

        num, cat = split_feature_types(df)

        assert len(cat) == 0
        assert len(num) == 2


# ======================
# BUILD PREPROCESSOR
# ======================


class TestBuildPreprocessor:

    def test_returns_column_transformer(self):
        """
        Given: listas de features
        When: build_preprocessor é chamado
        Then: retorna ColumnTransformer
        """
        numeric = ["age"]
        categorical = ["gender"]

        preprocessor = build_preprocessor(numeric, categorical)

        assert isinstance(preprocessor, ColumnTransformer)

    def test_preprocessor_transforms_data(self):
        """
        Given: dataframe simples
        When: fit_transform é aplicado
        Then: deve retornar array transformado
        """
        df = pd.DataFrame(
            {
                "age": [20, 30],
                "gender": ["M", "F"],
            }
        )

        preprocessor = build_preprocessor(["age"], ["gender"])

        result = preprocessor.fit_transform(df)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2


# ======================
# DROP ID COLUMNS
# ======================


class TestDropIdColumns:

    def test_removes_customer_id(self):
        """
        Given: dataframe com customerID
        When: drop_id_columns é chamado
        Then: coluna deve ser removida
        """
        df = pd.DataFrame(
            {
                "customerID": ["1", "2"],
                "age": [20, 30],
            }
        )

        result = drop_id_columns(df)

        assert "customerID" not in result.columns

    def test_does_nothing_if_no_id(self):
        """
        Given: dataframe sem customerID
        When: função é chamada
        Then: dataframe deve permanecer igual
        """
        df = pd.DataFrame(
            {
                "age": [20, 30],
            }
        )

        result = drop_id_columns(df)

        assert result.equals(df)
