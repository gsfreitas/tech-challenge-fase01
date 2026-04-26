# tests/test_example.py


def test_sample_data_has_correct_columns(sample_data):
    """
    Teste que verifica se a fixture sample_data possui
    as principais colunas esperadas do dataset de churn.
    """
    expected_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    for col in expected_columns:
        assert col in sample_data.columns


def test_sample_data_has_three_rows(sample_data):
    """
    Teste que verifica se a fixture sample_data possui
    exatamente 3 linhas.
    """
    assert len(sample_data) == 3


def test_sample_data_has_churn_column(sample_data):
    """
    Teste que verifica se a coluna alvo Churn existe.
    """
    assert "Churn" in sample_data.columns


def test_sample_data_churn_has_valid_values(sample_data):
    """
    Teste que verifica se a coluna Churn possui apenas
    valores válidos: Yes ou No.
    """
    valid_values = ["Yes", "No"]

    assert sample_data["Churn"].isin(valid_values).all()


def test_total_charges_starts_as_string(sample_data):
    """
    Teste que verifica se TotalCharges começa como texto,
    como acontece no dataset original.
    """
    assert sample_data["TotalCharges"].dtype == "object"
