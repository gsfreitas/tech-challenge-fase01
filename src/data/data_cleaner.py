import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Análise de missing values
        missing_values = self.df.isnull().sum()
        
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(self.df['TotalCharges'].median())
        
        self.df['MonthlyCharges'] = pd.to_numeric(self.df['MonthlyCharges'], errors='coerce')
        self.df['MonthlyCharges'] = self.df['MonthlyCharges'].fillna(self.df['MonthlyCharges'].median())

        # Se houver colunas, preencher com mediana
        for col in self.df.columns:
            if missing_values[col] > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
                    logging.info(f"Preenchidos {missing_values[col]} valores nulos na coluna '{col}' com a mediana ({median_value})")
                else:
                    mode_value = self.df[col].mode()[0]
                    self.df[col].fillna(mode_value, inplace=True)
                    logging.info(f"Preenchidos {missing_values[col]} valores nulos na coluna '{col}' com a moda ('{mode_value}')")
        
        logging.info("Limpeza de dados concluída. Nenhum valor nulo restante.")

        # Remove linhas duplicadas
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        logging.info(f"Removidos {initial_shape[0] - self.df.shape[0]} registros duplicados")

        # Remove linhas com valores nulos
        # initial_shape = self.df.shape
        # self.df.dropna(inplace=True)
        # logging.info(f"Removidos {initial_shape[0] - self.df.shape[0]} registros com valores nulos")

        return self.df