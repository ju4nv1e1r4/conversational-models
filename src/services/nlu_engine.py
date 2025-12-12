import os
import zipfile
import json
import logging
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.utils.storage import LocalStorage
from src.utils.config import settings

logger = logging.getLogger("nlu_engine")

class NLUEngine:
    def __init__(self, artifact_name: str, model_dir: str = "/app/models/served"):
        base_path = model_dir if model_dir else settings.ARTIFACTS_PATH
        self.storage = LocalStorage(base_path=base_path)
        self.artifact_name = artifact_name
        self.local_model_path = os.path.join(model_dir, artifact_name.replace(".zip", ""))
        self.session = None
        self.tokenizer = None

    def _load_artifacts(self):
        """Baixa (copia) do storage local e descompacta se necessário."""
        if os.path.exists(os.path.join(self.local_model_path, "model.onnx")):
            return

        logger.info(f"Instalando modelo {self.artifact_name}...")
        zip_local_path = f"/tmp/{self.artifact_name}"

        self.storage.download(self.artifact_name, zip_local_path)

        os.makedirs(self.local_model_path, exist_ok=True)
        with zipfile.ZipFile(zip_local_path, 'r') as zip_ref:
            zip_ref.extractall(self.local_model_path)
        
        os.remove(zip_local_path)

    def load(self):
        """Carrega ONNX Session e Tokenizer na memória RAM."""
        self._load_artifacts()

        tok_path = os.path.join(self.local_model_path, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1 # NOTE: to avoid excessive concurrency on cpus
        
        onnx_path = os.path.join(self.local_model_path, "model.onnx")
        self.session = ort.InferenceSession(onnx_path, sess_options)
        
        logger.info(f"Modelo {self.artifact_name} carregado na memória.")

    def predict(self, text: str, labels: list = None):
        """
        Exemplo genérico de inferência.
        Dependendo do modelo (ZeroShot, NER), a lógica de pre/post processing muda.
        """
        if not self.session:
            self.load()

        encoding = self.tokenizer.encode(text)
        
        input_feed = {
            self.session.get_inputs()[0].name: np.array([encoding.ids], dtype=np.int64),
            self.session.get_inputs()[1].name: np.array([encoding.attention_mask], dtype=np.int64)
        }

        if len(self.session.get_inputs()) > 2:
            input_feed[self.session.get_inputs()[2].name] = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self.session.run(None, input_feed)
        return outputs
