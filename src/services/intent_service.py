import numpy as np
import json
import logging

from src.services.nlu_engine import NLUEngine
from src.utils.telemetry import instrument

logger = logging.getLogger("intent_service")

class IntentService(NLUEngine):
    def __init__(self):
        super().__init__(artifact_name="intent_classifier.zip")
        self.entailment_id = None

    def _setup_tokenizer_config(self):
        """
        Configura padding e truncation explícitos para o tokenizer raw.
        Necessário para processamento em batch.
        """
        self.tokenizer.enable_truncation(max_length=512)

        pad_id = 0
        if self.tokenizer.token_to_id("[PAD]"):
            pad_id = self.tokenizer.token_to_id("[PAD]")
        elif self.tokenizer.token_to_id("<pad>"):
            pad_id = self.tokenizer.token_to_id("<pad>")
            
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

    def _get_entailment_id(self):
        """Descobre dinamicamente qual ID corresponde a 'entailment'."""
        if self.entailment_id is not None:
            return self.entailment_id

        config_path = f"{self.local_model_path}/config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            id2label = config.get("id2label", {})
            for id_str, label in id2label.items():
                if label.lower() == "entailment":
                    self.entailment_id = int(id_str)
                    return self.entailment_id

            self.entailment_id = 2 
            return 2
        except Exception as e:
            logger.error(f"Erro ao ler config.json: {e}")
            return 2

    @instrument(name="nlu_predict_intent")
    def predict_intent(self, text: str, candidate_labels: list[str]):
        """
        Realiza Zero-Shot Classification de forma dinâmica.
        Verifica quais inputs o modelo aceita antes de enviar.
        """
        if not self.session:
            self.load()
            self._setup_tokenizer_config()

        entailment_id = self._get_entailment_id()
        hypothesis_template = "This text is about {}."

        text_pairs = [(text, hypothesis_template.format(label)) for label in candidate_labels]

        encodings = self.tokenizer.encode_batch(text_pairs)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        model_input_names = [i.name for i in self.session.get_inputs()]
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if "token_type_ids" in model_input_names:
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            onnx_inputs["token_type_ids"] = token_type_ids

        logits = self.session.run(None, onnx_inputs)[0]
        
        entailment_logits = logits[:, entailment_id]

        exp_logits = np.exp(entailment_logits - np.max(entailment_logits))
        probs = exp_logits / exp_logits.sum()

        best_idx = np.argmax(probs)
        return candidate_labels[best_idx], float(probs[best_idx])
