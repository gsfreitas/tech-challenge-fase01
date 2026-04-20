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
MLRUNS_DIR = ROOT_DIR / "mlruns"

# Dataset e configurações globais
DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# MLflow
MLFLOW_EXPERIMENT_NAME = "tech-challenge-fase01"


def get_dataset_path() -> Path:
    """Retorna o caminho completo para o dataset."""
    return RAW_DATA_DIR / DATASET_NAME


def get_processed_data_path() -> Path:
    """Retorna o diretório de dados processados."""
    return PROCESSED_DATA_DIR


def get_models_dir() -> Path:
    """Retorna a pasta de modelos."""
    return MODELS_DIR


def get_mlruns_dir() -> Path:
    """Retorna a pasta de tracking do MLflow (raiz do projeto)."""
    return MLRUNS_DIR


def get_mlflow_tracking_uri() -> str:
    """Retorna o URI de tracking do MLflow como file:// path absoluto."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return MLRUNS_DIR.as_uri()