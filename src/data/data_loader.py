import logging
import csv
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class DataLoader:
    """
    Responsável por carregamento e validações do dataset
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.df: pd.DataFrame = None

    def _validate_csv_format(self) -> None:
        """Valida se o conteúdo do arquivo tem estrutura de CSV."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                csv.Sniffer().sniff(sample)
        except csv.Error as e:
            raise ValueError(f"O arquivo não parece ser um CSV válido: {e}")

    def load_data(self) -> pd.DataFrame:
        """
        Valida se o arquivo existe, se é um CSV e tenta carregar os dados.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        if self.file_path.suffix.lower() != '.csv':
            raise ValueError(f"O arquivo deve ser um CSV. Recebido: {self.file_path.suffix}")

        self._validate_csv_format()

        try:
            self.df = pd.read_csv(self.file_path)
            if self.df.empty:
                raise ValueError("O dataset está vazio.")
            logging.info(f"Dados carregados com sucesso: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")

        except Exception as e:
            logging.error(f"Erro ao carregar os dados: {e}")
            raise

        return self.df