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
    def __init__(self, artifact_name: str, runtime_dir: str = "/app/models/served"):
        """
        artifact_name: Nome do arquivo .zip (ex: intent_classifier.zip)
        runtime_dir: Onde o modelo será descompactado para rodar (Efêmero)
        """
        self.artifact_name = artifact_name
        self.storage = LocalStorage(base_path=settings.ARTIFACTS_PATH)
        self.runtime_dir = runtime_dir
        self.local_model_path = os.path.join(self.runtime_dir, artifact_name.replace(".zip", ""))
        self.session = None
        self.tokenizer = None

    def _load_artifacts(self):
        """Baixa do storage e descompacta no runtime dir."""
        if os.path.exists(os.path.join(self.local_model_path, "model.onnx")):
            return 

        logger.info(f"Instalando modelo {self.artifact_name} em {self.local_model_path}...")

        zip_local_path = f"/tmp/{self.artifact_name}"

        self.storage.download(self.artifact_name, zip_local_path)
        
        os.makedirs(self.local_model_path, exist_ok=True)

        with zipfile.ZipFile(zip_local_path, 'r') as zip_ref:
            zip_ref.extractall(self.local_model_path)

        os.remove(zip_local_path)

    def load(self):
        self._load_artifacts()

        tok_path = os.path.join(self.local_model_path, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1 
        
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
