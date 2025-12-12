import argparse
import logging
import sys
import os

sys.path.append(os.getcwd())

from src.services.model_builder import ModelBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cli_builder")

def main():
    parser = argparse.ArgumentParser(
        description="CLI para baixar modelos do HuggingFace, converter/empacotar e salvar no Storage Local."
    )

    parser.add_argument(
        "-m", "--model", 
        required=True, 
        help="ID do modelo no HuggingFace (ex: Xenova/nli-deberta-v3-xsmall)"
    )

    parser.add_argument(
        "-n", "--name", 
        required=True, 
        help="Nome do arquivo de saída .zip (ex: intent_classifier_v1.zip)"
    )

    args = parser.parse_args()

    logger.info(f"Iniciando build...")
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Artefato: {args.name}")

    try:
        builder = ModelBuilder(
            model_id=args.model,
            artifact_name=args.name
        )
        builder.run()
        logger.info("Processo finalizado com sucesso!")
        
    except Exception as e:
        logger.error(f"Falha no build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
