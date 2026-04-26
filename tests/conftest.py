# tests/conftest.py
# Fixtures compartilhadas para todos os testes.
# Fixtures definidas aqui ficam disponiveis automaticamente em todos os arquivos de teste.

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    # Fixture que fornece um dataframe de exemplo para testes.
    # Objetivo:
    # - Testar fluxo geral (pipeline completo)
    # - Testar transformação de dados categóricos
    # - Testar conversão de tipos (ex: TotalCharges como string)

    # Contém:
    # - Dados mistos (numéricos + categóricos)
    # - Estrutura semelhante ao dataset original

    # Returns:
    # pd.Dataframe: Dataset pequeno simulando dados.
    return pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
            "gender": ["Female", "Male", "Male"],
            "SeniorCitizen": [0, 0, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "No"],
            "tenure": [1, 34, 2],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No"],
            "InternetService": ["DSL", "DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes", "Yes"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["No", "No", "No"],
            "StreamingTV": ["No", "No", "No"],
            "StreamingMovies": ["No", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", "1889.5", "108.15"],
            "Churn": ["No", "No", "Yes"],
        }
    )


@pytest.fixture
def sample_data_with_missing_values():

    # Fixture com valores ausentes (missing values).

    # Objetivo:
    # - Testar tratamento de NaN
    # - Testar limpeza de dados (fillna, dropna, etc.)
    # - Testar conversão de valores inválidos (ex: string vazia)

    # Contém:
    # - None em colunas numéricas
    # - String vazia em TotalCharges
    # - Target com valor ausente

    return pd.DataFrame(
        {
            "customerID": ["001", "002", "003"],
            "tenure": [1, None, 12],
            "MonthlyCharges": [29.85, 56.95, None],
            "TotalCharges": ["29.85", " ", "300.00"],
            "Churn": ["No", "Yes", None],
        }
    )


@pytest.fixture
def sample_data_with_duplicates(sample_data: pd.DataFrame):
    # Fixture com dados duplicados.

    # Objetivo:
    # - Testar remoção de duplicatas
    # - Garantir que pipeline não quebra com dados repetidos

    # Estratégia:
    # - Duplica a primeira linha do dataset original
    duplicated_row = sample_data.iloc[[0]]
    return pd.concat([sample_data, duplicated_row], ignore_index=True)


@pytest.fixture
def sample_features():
    # Fixture com matriz numérica para testes de modelagem.

    # Objetivo:
    # - Testar normalização (scaler)
    # - Testar transformação de features
    # - Testar entrada de modelos

    # Formato:
    # - numpy array (simulando X)

    return np.array(
        [
            [5000, 34, 10000],
            [10000, 35, 50000],
            [30000, 39, 5000],
        ]
    )


@pytest.fixture
def categorical_columns():

    # Lista de colunas categóricas.

    # Objetivo:
    # - Testar encoding (LabelEncoder, OneHotEncoder)
    # - Separação entre tipos de dados

    # Usado em:
    # - Feature Engineering
    # - Pré-processamento

    return [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "Churn",
    ]


@pytest.fixture
def numerical_columns():

    # Lista de colunas numéricas.

    # Objetivo:
    # - Testar normalização
    # - Testar imputação
    # - Separação de features numéricas

    # Observação:
    # - TotalCharges começa como string e deve ser convertido

    return [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]


@pytest.fixture
def empty_dataframe():
    # Fixture com dataframe vazio para testar edge cases.

    return pd.DataFrame()
