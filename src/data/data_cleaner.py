import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class DataCleaner:
    """
    Responsável por limpeza e pré-processamento do dataset
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        """
        Realiza limpeza básica dos dados, como remoção de valores nulos e duplicados
        """
        
        # Remove linhas duplicadas
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        logging.info(f"Removidos {initial_shape[0] - self.df.shape[0]} registros duplicados")

        # Remove linhas com valores nulos
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        logging.info(f"Removidos {initial_shape[0] - self.df.shape[0]} registros com valores nulos")

        return self.df