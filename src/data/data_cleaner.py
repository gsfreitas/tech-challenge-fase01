import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Responsável por limpeza determinística do dataset, sem calcular
    estatísticas globais que possam contaminar o split train/test.

    Operações de imputação (mediana/moda) são responsabilidade do
    Pipeline sklearn (SimpleImputer), que é fitado apenas no fold de
    treino. Aqui só fazemos:

    - Conversão de tipos (strings numéricas -> float, com NaN onde falhar)
    - Remoção de duplicatas exatas
    - Log de qualidade dos dados

    Missing values permanecem como NaN e são tratados a jusante.
    """

    # Colunas que vêm como string no CSV mas deveriam ser numéricas
    NUMERIC_STRING_COLUMNS = ("TotalCharges", "MonthlyCharges")

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_data(self) -> pd.DataFrame:
        """
        Executa limpeza determinística do dataset.

        Retorna DataFrame com tipos corrigidos e duplicatas removidas.
        NaN permanecem e são tratados pelo pipeline sklearn.
        """
        self._coerce_numeric_columns()
        self._log_missing_values()
        self._drop_duplicates()

        logger.info(
            "Limpeza concluída: %s linhas, %s colunas, %s valores nulos restantes (tratados no pipeline)",
            self.df.shape[0],
            self.df.shape[1],
            self.df.isnull().sum().sum(),
        )

        return self.df

    def _coerce_numeric_columns(self) -> None:
        """Converte colunas numéricas em string para float, gerando NaN em falhas."""
        for col in self.NUMERIC_STRING_COLUMNS:
            if col not in self.df.columns:
                logger.warning("Coluna '%s' ausente; pulando conversão.", col)
                continue

            before_na = self.df[col].isnull().sum()
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            after_na = self.df[col].isnull().sum()

            new_na = after_na - before_na
            if new_na > 0:
                logger.info(
                    "Coluna '%s' convertida para numérico; %s valores coagidos para NaN.",
                    col,
                    new_na,
                )

    def _log_missing_values(self) -> None:
        """Apenas loga o panorama de missing values sem tratá-los."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            logger.info("Nenhum valor nulo encontrado após coerção de tipos.")
            return

        for col, count in missing.items():
            pct = 100 * count / len(self.df)
            logger.info(
                "Coluna '%s': %s valores nulos (%.2f%%) -- serão tratados no pipeline sklearn.",
                col,
                count,
                pct,
            )

    def _drop_duplicates(self) -> None:
        """Remove linhas duplicadas exatas."""
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        removed = initial_shape[0] - self.df.shape[0]
        if removed > 0:
            logger.info("Removidos %s registros duplicados.", removed)
        else:
            logger.info("Nenhum registro duplicado encontrado.")