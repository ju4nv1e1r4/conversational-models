import logging
import sys
import os

sys.path.append(os.getcwd())

from src.services.model_builder import ModelBuilder

logger = logging.getLogger("Entrypoint")
logging.basicConfig(level=logging.INFO)

MODEL_ID = "Xenova/nli-deberta-v3-xsmall"

def main():
    builder = ModelBuilder(
        model_id=MODEL_ID,
        artifact_name="intent_classifier.zip"
    )
    logger.info("Rodando builder...")
    if builder.run():
        logger.info("Builder finalizado.")
    else:
        logger.error("Builder falhou.")


if __name__ == "__main__":
    main()
