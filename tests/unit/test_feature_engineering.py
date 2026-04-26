# tests/unit/test_feature_engineering.py

import pandas as pd
import pytest

from src.data.data_cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineering:
    """Testes principais das transformações do FeatureEngineer."""

    # ======================
    # SUCCESS CASES
    # ======================

    def test_create_family_status(self, sample_data):
        """
        Given: DataFrame com Partner e Dependents
        When: create_family_status é chamado
        Then: coluna FamilyStatus deve ser criada corretamente
        """
        fe = FeatureEngineer(sample_data)

        result = fe.create_family_status()

        assert "FamilyStatus" in result.columns
        assert set(result["FamilyStatus"].unique()).issubset({"Single", "Family"})

    def test_create_tenure_bins(self, sample_data):
        """
        Given: DataFrame com coluna tenure
        When: create_tenure_bins é chamado
        Then: coluna TenureBin deve ser criada
        """
        fe = FeatureEngineer(sample_data)

        result = fe.create_tenure_bins()

        assert "TenureBin" in result.columns

    def test_create_family_size_proxy(self, sample_data):
        """
        Given: DataFrame com Partner e Dependents
        When: create_family_size_proxy é chamado
        Then: coluna FamilySizeProxy deve ser numérica
        """
        fe = FeatureEngineer(sample_data)

        result = fe.create_family_size_proxy()

        assert "FamilySizeProxy" in result.columns
        assert pd.api.types.is_numeric_dtype(result["FamilySizeProxy"])

    def test_create_risk_combo(self, sample_data):
        """
        Given: DataFrame com PaymentMethod e PaperlessBilling
        When: create_risk_combo é chamado
        Then: coluna RiskCombo deve ser criada
        """
        fe = FeatureEngineer(sample_data)

        result = fe.create_risk_combo()

        assert "RiskCombo" in result.columns
        assert set(result["RiskCombo"].unique()).issubset({"High Risk", "Low Risk"})

    def test_create_support_bundle(self, sample_data):
        """
        Given: DataFrame com TechSupport e OnlineSecurity
        When: create_support_bundle é chamado
        Then: coluna SupportBundle deve ser criada corretamente
        """
        fe = FeatureEngineer(sample_data)

        result = fe.create_support_bundle()

        assert "SupportBundle" in result.columns
        assert set(result["SupportBundle"].unique()).issubset(
            {"Has Bundle", "No Bundle"}
        )

    def test_create_diff_monthly_charges(self, sample_data):
        """
        Given: DataFrame limpo com TotalCharges numérico
        When: create_diff_monthly_charges é chamado
        Then: coluna DiffMonthlyCharges deve ser criada
        """
        clean_df = DataCleaner(sample_data).clean_data()
        fe = FeatureEngineer(clean_df)

        result = fe.create_diff_monthly_charges()

        assert "DiffMonthlyCharges" in result.columns
        assert pd.api.types.is_numeric_dtype(result["DiffMonthlyCharges"])

    def test_prepare_target(self, sample_data):
        """
        Given: DataFrame com coluna Churn
        When: prepare_target é chamado
        Then: Churn deve ser convertido para 0 e 1
        """
        fe = FeatureEngineer(sample_data)

        result = fe.prepare_target(sample_data)

        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_encode_categorical_features(self, sample_data):
        """
        Given: DataFrame com colunas categóricas
        When: encode_categorical_features é chamado
        Then: deve retornar DataFrame codificado com dummies
        """
        fe = FeatureEngineer(sample_data)

        result = fe.encode_categorical_features(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] != sample_data.shape[1]
        assert "gender" not in result.columns
        assert "Contract" not in result.columns

    # ======================
    # ERROR CASES
    # ======================

    def test_create_family_status_missing_columns(self):
        """
        Given: DataFrame sem Partner/Dependents
        When: create_family_status é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1, 2]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_family_status()

    def test_create_tenure_bins_missing_column(self):
        """
        Given: DataFrame sem tenure
        When: create_tenure_bins é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1, 2]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_tenure_bins()

    def test_create_family_size_proxy_missing_columns(self):
        """
        Given: DataFrame sem Partner/Dependents
        When: create_family_size_proxy é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1, 2]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_family_size_proxy()

    def test_create_risk_profile_missing_columns(self):
        """
        Given: DataFrame incompleto
        When: create_risk_profile é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"InternetService": ["DSL"]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_risk_profile()

    def test_create_risk_combo_missing_columns(self):
        """
        Given: DataFrame sem PaymentMethod/PaperlessBilling
        When: create_risk_combo é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1, 2]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_risk_combo()

    def test_create_support_bundle_missing_columns(self):
        """
        Given: DataFrame sem TechSupport/OnlineSecurity
        When: create_support_bundle é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1, 2]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_support_bundle()

    def test_create_diff_monthly_charges_missing_columns(self):
        """
        Given: DataFrame sem colunas necessárias
        When: create_diff_monthly_charges é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"MonthlyCharges": [10.0]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.create_diff_monthly_charges()

    def test_prepare_target_missing_column(self):
        """
        Given: DataFrame sem coluna alvo
        When: prepare_target é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.prepare_target(df)

    def test_encode_categorical_features_missing_columns(self):
        """
        Given: DataFrame incompleto
        When: encode_categorical_features é chamado
        Then: deve levantar ValueError
        """
        df = pd.DataFrame({"A": [1]})
        fe = FeatureEngineer(df)

        with pytest.raises(ValueError):
            fe.encode_categorical_features(df)

    def test_create_risk_profile(self):
        """
        Given: DataFrame com InternetService e serviços adicionais
        When: create_risk_profile é chamado
        Then: coluna RiskProfile deve ser criada corretamente
        """
        df = pd.DataFrame(
            {
                "InternetService": ["DSL", "Fiber optic", "No"],
                "OnlineSecurity": [0, 1, 0],
                "OnlineBackup": [0, 0, 0],
                "DeviceProtection": [0, 0, 0],
                "TechSupport": [0, 0, 0],
                "StreamingTV": [0, 0, 0],
                "StreamingMovies": [0, 0, 0],
            }
        )

        fe = FeatureEngineer(df)

        result = fe.create_risk_profile()

        assert "RiskProfile" in result.columns
        assert result["RiskProfile"].tolist() == ["High Risk", "Low Risk", "Low Risk"]
