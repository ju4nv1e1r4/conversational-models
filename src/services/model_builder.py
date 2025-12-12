import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

from src.utils.storage import LocalStorage
from src.utils.config import settings

logger = logging.getLogger("builder")
logging.basicConfig(level=logging.INFO)

class ModelBuilder:
    CONFIG_FILES = [
        "config.json", "spm.model", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json"
    ]

    CACHE_DIR = Path("/tmp/hf_cache")
    STAGING_DIR = Path("/tmp/staging")

    def __init__(self, model_id: str, artifact_name: str):
        self.model_id = model_id
        self.artifact_name = artifact_name
        self.storage = LocalStorage(base_path=settings.ARTIFACTS_PATH)

    def run(self):
        try:
            logger.info(f"Baixando snapshot de {self.model_id}...")
            clone_dir = snapshot_download(
                repo_id=self.model_id,
                local_dir_use_symlinks=False,
                cache_dir=self.CACHE_DIR,
                allow_patterns=["*.json", "*.onnx", "*.model"] # NOTE: allow_patterns: download only importants stuffs for inferences
            )
            clone_path = Path(clone_dir)

            if self.STAGING_DIR.exists():
                shutil.rmtree(self.STAGING_DIR)
            self.STAGING_DIR.mkdir(parents=True)

            for filename in self.CONFIG_FILES:
                src = clone_path / filename
                if src.exists():
                    shutil.copy(src, self.STAGING_DIR)

            onnx_files = list(clone_path.glob('**/*.onnx'))
            if not onnx_files:
                raise FileNotFoundError("Nenhum .onnx encontrado!")

            shutil.copy(onnx_files[0], self.STAGING_DIR / "model.onnx")

            zip_name = self.artifact_name.replace(".zip", "")
            zip_path = shutil.make_archive(f"/tmp/{zip_name}", 'zip', self.STAGING_DIR)

            self.storage.upload(zip_path, self.artifact_name)
            
            logger.info(f"Sucesso! Artefato {self.artifact_name} criado.")
            
        finally:
            # cleanup
            if self.STAGING_DIR.exists():
                shutil.rmtree(self.STAGING_DIR)
