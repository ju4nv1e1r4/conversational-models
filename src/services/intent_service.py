import numpy as np
from src.services.nlu_engine import NLUEngine

class IntentService(NLUEngine):
    def __init__(self):
        super().__init__(artifact_name="intent_classifier.zip")
        self.labels_map = {}

    def _get_entailment_id(self):
        """Descobre qual ID corresponde a 'entailment' no config do modelo."""
        if not self.session:
            self.load()
            
        config_path = f"{self.local_model_path}/config.json"
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        for id_str, label in id2label.items():
            if label.lower() == "entailment":
                return int(id_str)
        return 2 # deberta default fallback

    def predict_intent(self, text: str, candidate_labels: list[str]):
        """
        Realiza Zero-Shot Classification.
        Retorna: (melhor_label, score)
        """
        if not self.session:
            self.load()

        entailment_id = self._get_entailment_id()
        scores = []

        hypothesis_template = "This text is about {}." 

        for label in candidate_labels:
            hypothesis = hypothesis_template.format(label)

            inputs = self.tokenizer(text, hypothesis, return_tensors="np", padding=True, truncation=True)
            
            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"]

            logits = self.session.run(None, onnx_inputs)[0][0]

            entailment_score = logits[entailment_id]
            scores.append(entailment_score)

        scores_np = np.array(scores)
        exp_scores = np.exp(scores_np - np.max(scores_np))
        probs = exp_scores / exp_scores.sum()

        best_idx = np.argmax(probs)
        return candidate_labels[best_idx], float(probs[best_idx])