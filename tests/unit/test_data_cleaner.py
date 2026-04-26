# tests/unit/test_data_cleaner.py

import pandas as pd

from src.data.data_cleaner import DataCleaner


class TestDataCleaner:
    """Testes principais de comportamento do DataCleaner."""

    # ======================
    # SUCCESS CASES
    # ======================

    def test_clean_data_returns_dataframe(self, sample_data):
        """
        Given: um DataFrame válido
        When: clean_data é chamado
        Then: deve retornar um pandas DataFrame
        """
        cleaner = DataCleaner(sample_data)

        result = cleaner.clean_data()

        assert isinstance(result, pd.DataFrame)

    def test_clean_data_converts_total_charges_to_numeric(self, sample_data):
        """
        Given: um DataFrame com TotalCharges como string numérica
        When: clean_data é chamado
        Then: TotalCharges deve ser convertida para tipo numérico
        """
        cleaner = DataCleaner(sample_data)

        result = cleaner.clean_data()

        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_clean_data_converts_monthly_charges_to_numeric(self, sample_data):
        """
        Given: um DataFrame com MonthlyCharges numérico
        When: clean_data é chamado
        Then: MonthlyCharges deve permanecer como tipo numérico
        """
        cleaner = DataCleaner(sample_data)

        result = cleaner.clean_data()

        assert pd.api.types.is_numeric_dtype(result["MonthlyCharges"])

    def test_clean_data_removes_duplicates(self, sample_data_with_duplicates):
        """
        Given: um DataFrame com linhas duplicadas
        When: clean_data é chamado
        Then: as duplicatas exatas devem ser removidas
        """
        cleaner = DataCleaner(sample_data_with_duplicates)

        result = cleaner.clean_data()

        assert len(result) == 3
        assert result.duplicated().sum() == 0

    def test_clean_data_keeps_missing_values_as_nan(
        self, sample_data_with_missing_values
    ):
        """
        Given: um DataFrame com valores ausentes
        When: clean_data é chamado
        Then: valores ausentes devem permanecer como NaN
        """
        cleaner = DataCleaner(sample_data_with_missing_values)

        result = cleaner.clean_data()

        assert result.isnull().sum().sum() > 0

    def test_clean_data_converts_invalid_numeric_strings_to_nan(self):
        """
        Given: um DataFrame com strings inválidas em colunas numéricas
        When: clean_data é chamado
        Then: valores inválidos devem ser convertidos para NaN
        """
        df = pd.DataFrame(
            {
                "TotalCharges": ["100.50", "valor_invalido", " "],
                "MonthlyCharges": ["29.85", "erro", "56.95"],
            }
        )

        cleaner = DataCleaner(df)

        result = cleaner.clean_data()

        assert result["TotalCharges"].isnull().sum() == 2
        assert result["MonthlyCharges"].isnull().sum() == 1

    def test_clean_data_does_not_modify_original_dataframe(self, sample_data):
        """
        Given: um DataFrame original
        When: DataCleaner é instanciado e clean_data é chamado
        Then: o DataFrame original não deve ser alterado diretamente
        """
        original = sample_data.copy(deep=True)

        cleaner = DataCleaner(sample_data)
        cleaner.clean_data()

        assert sample_data.equals(original)

    # ======================
    # EDGE CASES
    # ======================

    def test_clean_data_handles_missing_numeric_columns(self):
        """
        Given: um DataFrame sem TotalCharges e MonthlyCharges
        When: clean_data é chamado
        Then: a limpeza deve executar sem erro
        """
        df = pd.DataFrame(
            {
                "customerID": ["001", "002"],
                "Churn": ["No", "Yes"],
            }
        )

        cleaner = DataCleaner(df)

        result = cleaner.clean_data()

        assert list(result.columns) == ["customerID", "Churn"]
        assert len(result) == 2

    def test_clean_data_handles_empty_dataframe(self, empty_dataframe):
        """
        Given: um DataFrame vazio
        When: clean_data é chamado
        Then: deve retornar um DataFrame vazio sem erro
        """
        cleaner = DataCleaner(empty_dataframe)

        result = cleaner.clean_data()

        assert result.empty


class TestDataCleanerWithMock:
    """Testes que isolam logs e métodos internos do DataCleaner usando mock."""

    # ======================
    # MOCK BEHAVIOR
    # ======================

    def test_clean_data_calls_internal_methods(self, mocker, sample_data):
        """
        Given: um DataCleaner com DataFrame válido
        When: clean_data é chamado
        Then: os métodos internos de limpeza devem ser chamados uma vez
        """
        cleaner = DataCleaner(sample_data)

        mock_coerce = mocker.patch.object(cleaner, "_coerce_numeric_columns")
        mock_log_missing = mocker.patch.object(cleaner, "_log_missing_values")
        mock_drop_duplicates = mocker.patch.object(cleaner, "_drop_duplicates")

        cleaner.clean_data()

        mock_coerce.assert_called_once()
        mock_log_missing.assert_called_once()
        mock_drop_duplicates.assert_called_once()

    def test_coerce_numeric_columns_logs_warning_when_column_missing(self, mocker):
        """
        Given: um DataFrame sem colunas numéricas esperadas
        When: _coerce_numeric_columns é chamado
        Then: logger.warning deve ser chamado para cada coluna ausente
        """
        df = pd.DataFrame({"customerID": ["001"]})
        cleaner = DataCleaner(df)

        mock_warning = mocker.patch("src.data.data_cleaner.logger.warning")

        cleaner._coerce_numeric_columns()

        assert mock_warning.call_count == 2

    def test_coerce_numeric_columns_logs_info_when_values_become_nan(self, mocker):
        """
        Given: um DataFrame com valor inválido em coluna numérica
        When: _coerce_numeric_columns é chamado
        Then: logger.info deve registrar valores convertidos para NaN
        """
        df = pd.DataFrame(
            {
                "TotalCharges": ["100", "erro"],
                "MonthlyCharges": ["20", "30"],
            }
        )
        cleaner = DataCleaner(df)

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner._coerce_numeric_columns()

        mock_info.assert_called_once()

    def test_log_missing_values_logs_when_no_missing_values(self, mocker, sample_data):
        """
        Given: um DataFrame sem valores nulos
        When: _log_missing_values é chamado
        Then: logger.info deve informar que não há valores nulos
        """
        cleaner = DataCleaner(sample_data)
        cleaner._coerce_numeric_columns()

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner._log_missing_values()

        mock_info.assert_called_once_with(
            "Nenhum valor nulo encontrado após coerção de tipos."
        )

    def test_log_missing_values_logs_each_missing_column(
        self, mocker, sample_data_with_missing_values
    ):
        """
        Given: um DataFrame com valores nulos em múltiplas colunas
        When: _log_missing_values é chamado
        Then: logger.info deve ser chamado uma vez para cada coluna com nulos
        """
        cleaner = DataCleaner(sample_data_with_missing_values)
        cleaner._coerce_numeric_columns()

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner._log_missing_values()

        assert mock_info.call_count == 4

    def test_drop_duplicates_logs_when_duplicates_removed(
        self, mocker, sample_data_with_duplicates
    ):
        """
        Given: um DataFrame com duplicatas
        When: _drop_duplicates é chamado
        Then: logger.info deve registrar a quantidade removida
        """
        cleaner = DataCleaner(sample_data_with_duplicates)

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner._drop_duplicates()

        mock_info.assert_called_once_with(
            "Removidos %s registros duplicados.",
            1,
        )

    def test_drop_duplicates_logs_when_no_duplicates(self, mocker, sample_data):
        """
        Given: um DataFrame sem duplicatas
        When: _drop_duplicates é chamado
        Then: logger.info deve informar que não há duplicatas
        """
        cleaner = DataCleaner(sample_data)

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner._drop_duplicates()

        mock_info.assert_called_once_with("Nenhum registro duplicado encontrado.")

    def test_clean_data_logs_final_summary(self, mocker, sample_data):
        """
        Given: um DataFrame válido
        When: clean_data é chamado
        Then: logger.info deve registrar o resumo final da limpeza
        """
        cleaner = DataCleaner(sample_data)

        mock_info = mocker.patch("src.data.data_cleaner.logger.info")

        cleaner.clean_data()

        assert mock_info.call_count >= 1
