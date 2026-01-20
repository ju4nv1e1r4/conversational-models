import logging
import sys
import os

sys.path.append(os.getcwd())

from src.services.model_builder import ModelBuilder
from src.utils.config import settings

logger = logging.getLogger("Entrypoint")
logging.basicConfig(level=logging.INFO)

MODEL_ID = "Xenova/nli-deberta-v3-xsmall"
ARTIFACT_NAME = "intent_classifier.zip"

def main():
    artifact_path = os.path.join(settings.ARTIFACTS_PATH, ARTIFACT_NAME)
    if os.path.exists(artifact_path):
        logger.info(f"Modelo já existe em {artifact_path}. Pulando build.")
        return

    builder = ModelBuilder(
        model_id=MODEL_ID,
        artifact_name=ARTIFACT_NAME
    )
    logger.info("Rodando builder...")
    if builder.run():
        logger.info("Builder finalizado.")
    else:
        logger.error("Builder falhou.")


if __name__ == "__main__":
    main()
