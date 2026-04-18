import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


# Para usar, apenas colcoar no scripts
# from src.utils.logging_config import setup_logging

# setup_logging()
