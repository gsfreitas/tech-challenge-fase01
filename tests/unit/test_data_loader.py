# tests/unit/test_data_loader.py

# Testes unitaios para o modulo DataLoader.
# Estes testes verificam o comportamento das funcoes de
# carregamento e preprocessamento de dados de churn

import csv

import pandas as pd
import pytest

from src.data.data_loader import DataLoader


class TestDataLoader:
    """
    Testes unitários para a classe DataLoader.
    """

    # ======================
    # SUCCESS CASES
    # ======================

    def test_load_data_returns_dataframe(self, tmp_path):
        # Verifica que load_data retorna um DataFrame.
        """
        Given: um arquivo CSV válido com dados
        When: o método load_data é chamado
        Then: deve retornar um pandas DataFrame
        """
        #  Arrange
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        loader = DataLoader(csv_file)
        #  Act
        result = loader.load_data()

        assert isinstance(result, pd.DataFrame)

    def test_load_data_has_expected_rows(self, tmp_path):
        """
        Given: um arquivo CSV válido com duas linhas de dados
        When: o método load_data é chamado
        Then: deve retornar um DataFrame com duas linhas
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n" "002,Yes\n",
            encoding="utf-8",
        )

        loader = DataLoader(csv_file)

        result = loader.load_data()

        assert len(result) == 2

    def test_load_data_has_expected_columns(self, tmp_path):
        """
        Given: um arquivo CSV válido com colunas do dataset de churn
        When: o método load_data é chamado
        Then: o DataFrame deve conter as colunas esperadas
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,gender,tenure,MonthlyCharges,TotalCharges,Churn\n"
            "7590-VHVEG,Female,1,29.85,29.85,No\n",
            encoding="utf-8",
        )

        loader = DataLoader(csv_file)

        result = loader.load_data()

        expected_columns = [
            "customerID",
            "gender",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_load_data_saves_dataframe_in_loader(self, tmp_path):
        """
        Given: um arquivo CSV válido
        When: o método load_data é chamado
        Then: o DataFrame carregado deve ser salvo no atributo df do loader
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        loader = DataLoader(csv_file)

        df = loader.load_data()

        assert loader.df is not None
        assert loader.df.equals(df)

    # ======================
    # ERROR CASES
    # ======================

    def test_load_data_file_not_found(self, tmp_path):
        """
        Given: um caminho para um arquivo inexistente
        When: o método load_data é chamado
        Then: deve levantar FileNotFoundError
        """
        file_path = tmp_path / "nao_existe.csv"

        loader = DataLoader(file_path)

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_load_data_wrong_extension(self, tmp_path):
        """
        Given: um arquivo com extensão diferente de .csv
        When: o método load_data é chamado
        Then: deve levantar ValueError informando que o arquivo deve ser CSV
        """
        file = tmp_path / "data.txt"
        file.write_text("customerID,Churn\n001,No", encoding="utf-8")

        loader = DataLoader(file)

        with pytest.raises(ValueError, match="CSV"):
            loader.load_data()

    def test_load_data_empty_dataset(self, tmp_path):
        """
        Given: um arquivo CSV vazio
        When: o método load_data é chamado
        Then: deve levantar uma exceção indicando problema no carregamento
        """
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        loader = DataLoader(csv_file)

        with pytest.raises(Exception):
            loader.load_data()

    def test_load_data_csv_with_only_header(self, tmp_path):
        """
        Given: um arquivo CSV com cabeçalho, mas sem linhas de dados
        When: o método load_data é chamado
        Then: deve levantar ValueError informando que o dataset está vazio
        """
        csv_file = tmp_path / "only_header.csv"
        csv_file.write_text(
            "customerID,gender,tenure,MonthlyCharges,TotalCharges,Churn\n",
            encoding="utf-8",
        )

        loader = DataLoader(csv_file)

        with pytest.raises(ValueError, match="O dataset está vazio"):
            loader.load_data()

    def test_invalid_csv_format(self, mocker, tmp_path):
        """
        Given: um arquivo com extensão .csv, mas formato inválido
        When: o método load_data é chamado e a validação CSV falha
        Then: deve levantar ValueError informando que o CSV não é válido
        """
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("lixo", encoding="utf-8")

        mocker.patch(
            "src.data.data_loader.csv.Sniffer.sniff",
            side_effect=csv.Error("erro"),
        )

        loader = DataLoader(csv_file)

        with pytest.raises(ValueError, match="CSV válido"):
            loader.load_data()


class TestDataLoaderWithMock:
    """Testes que isolam dependências externas do DataLoader usando mock."""

    # ======================
    # MOCK BEHAVIOR
    # ======================

    def test_load_data_calls_read_csv(self, mocker, tmp_path):
        """
        Given: um arquivo CSV existente e válido
        When: load_data é chamado
        Then: pd.read_csv deve ser chamado uma vez com o caminho do arquivo
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        mock_read_csv = mocker.patch(
            "src.data.data_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "customerID": ["001"],
                    "Churn": ["No"],
                }
            ),
        )

        loader = DataLoader(csv_file)

        result = loader.load_data()

        mock_read_csv.assert_called_once_with(csv_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_load_data_calls_csv_sniffer(self, mocker, tmp_path):
        """
        Given: um arquivo CSV existente
        When: load_data é chamado
        Then: csv.Sniffer().sniff deve ser chamado para validar o formato
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        mock_sniff = mocker.patch(
            "src.data.data_loader.csv.Sniffer.sniff",
            return_value=None,
        )

        loader = DataLoader(csv_file)

        loader.load_data()

        mock_sniff.assert_called_once()

    def test_load_data_raises_when_read_csv_fails(self, mocker, tmp_path):
        """
        Given: um arquivo CSV existente
        When: pd.read_csv falha durante o carregamento
        Then: a exceção original deve ser propagada
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        mocker.patch(
            "src.data.data_loader.pd.read_csv",
            side_effect=pd.errors.ParserError("erro ao ler csv"),
        )

        loader = DataLoader(csv_file)

        with pytest.raises(pd.errors.ParserError):
            loader.load_data()

    def test_load_data_logs_success_when_loaded(self, mocker, tmp_path):
        """
        Given: um arquivo CSV válido
        When: load_data carrega os dados com sucesso
        Then: logger.info deve ser chamado uma vez
        """
        csv_file = tmp_path / "churn.csv"
        csv_file.write_text(
            "customerID,Churn\n" "001,No\n",
            encoding="utf-8",
        )

        mock_logger_info = mocker.patch("src.data.data_loader.logger.info")

        loader = DataLoader(csv_file)

        loader.load_data()

        mock_logger_info.assert_called_once()
