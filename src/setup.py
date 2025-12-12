from src.services.model_builder import ModelBuilder

MODEL_ID = "Xenova/nli-deberta-v3-xsmall"

def main():
    builder = ModelBuilder(
        model_id=MODEL_ID,
        artifact_name="intent_classifier_v1.zip"
    )
    builder.run()

if __name__ == "__main__":
    main()