import logging
import pipeline

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("=== Iniciando pipeline... ===")
    pipeline.main()
    logging.info("=== Pipeline finalizado. ===")