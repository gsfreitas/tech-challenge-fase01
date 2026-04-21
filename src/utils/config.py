from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
MLRUNS_DIR = ROOT_DIR / "mlruns"

DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# hiperparametros do modelo
MLP_HIDDEN_DIMS = (128, 64, 32)
MLP_DROPOUT_RATES = (0.3, 0.3, 0.2)
MLP_LEARNING_RATE = 1e-3
MLP_BATCH_SIZE = 64
MLP_MAX_EPOCHS = 100
MLP_EARLY_STOPPING_PATIENCE = 10
MLP_VAL_SIZE = 0.15

MLFLOW_EXPERIMENT_NAME = "tech-challenge-fase01"


def get_dataset_path() -> Path:
    """
    Retorna o caminho completo para o dataset
    """
    return RAW_DATA_DIR / DATASET_NAME


def get_processed_data_path() -> Path:
    """
    Retorna o diretório de dados processados
    """
    return PROCESSED_DATA_DIR


def get_models_dir() -> Path:
    """
    Retorna a pasta de modelos
    """
    return MODELS_DIR


def get_mlruns_dir() -> Path:
    """
    Retorna a pasta de tracking do MLflow (raiz do projeto)
    """
    return MLRUNS_DIR


def get_mlflow_tracking_uri() -> str:
    """
    Retorna o URI de tracking do MLflow como file:// path absoluto.
    """
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return MLRUNS_DIR.as_uri()