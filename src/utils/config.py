from pathlib import Path

def get_dataset_path() -> Path:
    """Retorna o caminho completo para o dataset."""
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    # Caminhos do dataset
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "ibm-telco-customer-churn.csv"
    
    return RAW_DATA_PATH

def get_processed_data_path() -> Path:
    """Retorna o caminho completo para o dataset."""
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    # Caminhos do dataset
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DATA_PATH = DATA_DIR / "processed"
    
    return PROCESSED_DATA_PATH