"""
Pipeline principal do projeto, orquestrando as etapas de 
carregamento, limpeza e engenharia de features.
"""

from utils.config import get_dataset_path, get_processed_data_path
from data.data_loader import DataLoader
from data.data_cleaner import DataCleaner
from features.feature_engineering import FeatureEngineer


def main():
    # Etapa 1: Carregamento dos dados
    data_loader = DataLoader(file_path=get_dataset_path())  # ← chamada com ()

    # Etapa 2: Limpeza dos dados
    data_cleaner = DataCleaner(df=data_loader.load_data())  # ← parâmetro correto: df
    df_cleaned = data_cleaner.clean_data()

    # Etapa 3: Engenharia de features
    feature_engineer = FeatureEngineer(df=df_cleaned)
    df_cleaned = feature_engineer.create_family_status()  # ← sem df=, usa self.df
    df_cleaned = feature_engineer.create_tenure_bins()  # ← sem df
    df_cleaned = feature_engineer.create_family_size_proxy()  # ← sem df=
    df_cleaned = feature_engineer.create_risk_profile()  # ← sem df=
    df_cleaned = feature_engineer.create_risk_combo()  # ← sem df=
    df_cleaned = feature_engineer.create_support_bundle()  # ← sem df=
    df_cleaned = feature_engineer.create_diff_monthly_charges()  # ← sem df=
    df_encoded = feature_engineer.encode_categorical_features(df_cleaned)  # ← sem df=, usa self.df

    feature_engineer.prepare_target(df=df_encoded)

    # insere o arquivo modificado em data/processed para futuras etapas de modelagem
    df_encoded.to_csv(get_processed_data_path() / "processed_customer_churn.csv", index=False)

    print("Pipeline concluído com sucesso. DataFrame final pronto para modelagem.")