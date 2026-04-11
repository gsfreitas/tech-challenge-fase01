import pandas as pd
import logging

class DataLoader:
    """
    Responsável pelo carregamento e limpeza inicial do dataset IBM Telco.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Carrega o CSV e armazena no estado da instância."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dados carregados com sucesso: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Erro ao carregar arquivo: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """Executa a limpeza técnica (conversão de tipos e tratamento de nulos)."""
        if self.df is None:
            raise ValueError("O dataframe não foi carregado. Chame load_data() primeiro.")
        
        # Correção do campo TotalCharges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)
        
        print("Limpeza técnica concluída.")
        return self.df