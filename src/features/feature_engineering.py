import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding nas variáveis categóricas relevantes.
        Substitui a lógica manual do notebook por uma função modular.
        """
        
        categorical_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
            "Contract", "PaperlessBilling", "PaymentMethod"
        ]

        # avalia se o dataset possui as colunas
        missing_cols = [col for col in categorical_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Colunas faltando para encoding: {missing_cols}")
            raise ValueError(f"Colunas faltando para encoding: {missing_cols}")
        else:
            logging.info("Todas as colunas necessárias para encoding estão presentes.")
            return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    def prepare_target(self, df: pd.DataFrame, target_col='Churn') -> pd.DataFrame:
        """
        Converte a variável alvo para binário (0 e 1).
        """
        if target_col not in df.columns:
            logging.error(f"Coluna alvo '{target_col}' não encontrada no dataframe.")
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataframe.")
        
        df_final = df.copy()
        df_final[target_col] = df_final[target_col].map({'Yes': 1, 'No': 0})
        logging.info(f"Variável alvo '{target_col}' convertida para binário com sucesso.")
        return df_final