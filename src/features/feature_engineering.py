import pandas as pd
import logging
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def create_family_status(self) -> pd.DataFrame:
        """
        Cria a variável 'FamilyStatus' combinando 'Partner' e 'Dependents'
        """

        # avalia se as colunas existem
        if 'Partner' not in self.df.columns or 'Dependents' not in self.df.columns:
            logging.error("Colunas 'Partner' ou 'Dependents' não encontradas.")
            raise ValueError("Colunas 'Partner' ou 'Dependents' não encontradas.")
        else:
            logging.info("Colunas 'Partner' e 'Dependents' encontradas. Criando 'FamilyStatus'.")
            self.df['FamilyStatus'] = self.df.apply(
                lambda row: 'Single' if row['Partner'] == 'No' and row['Dependents'] == 'No' else 'Family', 
                axis=1
            )
            logging.info("Variável 'FamilyStatus' criada com sucesso.")
            return self.df
        
    def create_tenure_bins(self) -> pd.DataFrame:
        """
        Cria a variável 'TenureBin' categorizando a variável 'tenure' em bins.
        """

        if 'tenure' not in self.df.columns:
            logging.error("Coluna 'tenure' não encontrada.")
            raise ValueError("Coluna 'tenure' não encontrada.")
        else:
            logging.info("Coluna 'tenure' encontrada. Criando 'TenureBin'.")
            bins = [0, 12, 24, 48, 60, float('inf')]
            labels = ['0-12', '13-24', '25-48', '49-60', '61+']
            self.df['TenureBin'] = pd.cut(self.df['tenure'], bins=bins, labels=labels)
            logging.info("Variável 'TenureBin' criada com sucesso.")
            return self.df
        
    def create_family_size_proxy(self) -> pd.DataFrame:
        """
        Cria a variável FamilySizeProxy como uma proxy para o tamanho da família, combinando Partner e Dependents
        """
        
        if 'Partner' not in self.df.columns or 'Dependents' not in self.df.columns:
            logging.error("Colunas 'Partner' ou 'Dependents' não encontradas.")
            raise ValueError("Colunas 'Partner' ou 'Dependents' não encontradas.")
        else:
            logging.info("Colunas 'Partner' e 'Dependents' encontradas. Criando 'FamilySizeProxy'.")
            
            self.df['FamilySizeProxy'] = self.df.apply(
                lambda row: 1 + (row['Partner'] == 'Yes') + (row['Dependents'] == 'Yes'), 
                axis=1
            )
            logging.info("Variável 'FamilySizeProxy' criada com sucesso.")
            return self.df

    def create_risk_profile(self) -> pd.DataFrame:
        """
        Cria o RiskProfile para clientes que possuem Internet sem nenhum serviço adicional (perfil de risco)
        """

        internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        missing_cols = [col for col in internet_services + ['InternetService'] if col not in self.df.columns]
        if missing_cols:
            logging.error(f"Colunas faltando para criação de 'RiskProfile': {missing_cols}")
            raise ValueError(f"Colunas faltando para criação de 'RiskProfile': {missing_cols}")
        else:
            logging.info("Todas as colunas necessárias para criar 'RiskProfile' estão presentes.")
            self.df["RiskProfile"] = np.where((self.df['InternetService'] != 'No') & (self.df[internet_services].sum(axis=1) == 0), 1, 0)
            self.df["RiskProfile"] = self.df['RiskProfile'].map({1: 'High Risk', 0: 'Low Risk'})
            logging.info("Variável 'RiskProfile' criada com sucesso.")
            return self.df
        
    def create_risk_combo(self) -> pd.DataFrame:
        """
        Combinação de risco: boleto eletrônico + sem papel
        """

        if 'PaymentMethod' not in self.df.columns or 'PaperlessBilling' not in self.df.columns:
            logging.error("Colunas 'PaymentMethod' ou 'PaperlessBilling' não encontradas.")
            raise ValueError("Colunas 'PaymentMethod' ou 'PaperlessBilling' não encontradas.")
        else:
            logging.info("Colunas 'PaymentMethod' e 'PaperlessBilling' encontradas. Criando 'RiskCombo'.")
            self.df['RiskCombo'] = np.where(
                (self.df['PaymentMethod'] == 'Electronic check') & (self.df['PaperlessBilling'] == 'Yes'), 1, 0)
            self.df['RiskCombo'] = self.df['RiskCombo'].map({1: 'High Risk', 0: 'Low Risk'})
            logging.info("Variável 'RiskCombo' criada com sucesso.")
            return self.df
        
    def create_support_bundle(self) -> pd.DataFrame:
        """
        Cria a variável 'SupportBundle' para clientes que possuem TechSupport e OnlineSecurity (pacote de suporte)
        """

        if 'TechSupport' not in self.df.columns or 'OnlineSecurity' not in self.df.columns:
            logging.error("Colunas 'TechSupport' ou 'OnlineSecurity' não encontradas.")
            raise ValueError("Colunas 'TechSupport' ou 'OnlineSecurity' não encontradas.")
        else:
            logging.info("Colunas 'TechSupport' e 'OnlineSecurity' encontradas. Criando 'SupportBundle'.")
            self.df['SupportBundle'] = np.where((self.df['TechSupport'] == 'Yes') & (self.df['OnlineSecurity'] == 'Yes'), 1, 0)
            self.df['SupportBundle'] = self.df['SupportBundle'].map({1: 'Has Bundle', 0: 'No Bundle'})
            logging.info("Variável 'SupportBundle' criada com sucesso.")
            return self.df

    def create_diff_monthly_charges(self) -> pd.DataFrame:
        """
        Cria a variável 'DiffMonthlyCharges' calculando a diferença entre 'MonthlyCharges' e 'TotalCharges' dividida por 'tenure'.
        Calula a diferença entre o valor atual e o valor médio histórico
        Pode indicar se houve upgrade, downgrade de plano ou mudança no consumo
        """

        required_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            logging.error(f"Colunas faltando para criação de 'DiffMonthlyCharges': {missing_cols}")
            raise ValueError(f"Colunas faltando para criação de 'DiffMonthlyCharges': {missing_cols}")
        else:
            logging.info("Todas as colunas necessárias para criar 'DiffMonthlyCharges' estão presentes.")
            self.df['DiffMonthlyCharges'] = (self.df['MonthlyCharges'] - self.df['TotalCharges'] / self.df['tenure']).fillna(0)
            logging.info("Variável 'DiffMonthlyCharges' criada com sucesso.")
            return self.df

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
            
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logging.info(f"Shape do dataframe depois do encoding: {df_encoded.shape}")
            return df_encoded
    
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