from pathlib import Path

# Raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[2]

# Diretórios principais
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "src" / "models"
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Dataset e configurações globais
DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def get_dataset_path() -> Path:
    """Retorna o caminho completo para o dataset."""
    return RAW_DATA_DIR / DATASET_NAME


def get_processed_data_path() -> Path:
    """Retorna o caminho completo para o dataset."""
    return PROCESSED_DATA_DIR


def get_models_dir() -> Path:
    """Retorna a pasta de modelos."""
    return MODELS_DIR
