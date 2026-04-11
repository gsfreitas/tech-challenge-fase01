import pandas as pd
import logging

class DataCleaner:
    """
    Responsável pela limpeza dos dados.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def clean_data(self) -> pd.DataFrame:
        """Executa a limpeza técnica (conversão de tipos e tratamento de nulos)."""
        if self.df is None:
            raise ValueError("O dataframe não foi carregado!")
        
        # Correção do campo TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)

        

        print("Limpeza técnica concluída.")
        return self.df