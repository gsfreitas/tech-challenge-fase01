from src.features.feature_engineering import FeatureEngineer
import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(df)

    df = fe.create_family_status()
    df = fe.create_tenure_bins()
    df = fe.create_family_size_proxy()
    df = fe.create_risk_profile()
    df = fe.create_risk_combo()
    df = fe.create_support_bundle()
    df = fe.create_diff_monthly_charges()

    return df

