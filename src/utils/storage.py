import shutil
import os
import logging

logger = logging.getLogger("storage")

class LocalStorage:
    """
    Simula um Bucket de Cloud Service, mas operando no sistema de arquivos local.
    Útil para desenvolvimento e ambientes air-gapped.
    """
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def upload(self, local_file: str, remote_path: str):
        """Copia arquivo local para o 'bucket' local."""
        dest_path = os.path.join(self.base_path, remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(local_file, dest_path)
        logger.info(f"Upload (Copy): {local_file} -> {dest_path}")

    def download(self, remote_path: str, local_dest: str):
        """Copia do 'bucket' local para o destino."""
        src_path = os.path.join(self.base_path, remote_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Objeto não encontrado no storage: {src_path}")
        
        os.makedirs(os.path.dirname(local_dest), exist_ok=True)
        shutil.copy2(src_path, local_dest)
        logger.info(f"Download (Copy): {src_path} -> {local_dest}")

    def list(self, directory: str = ""):
        """Lista arquivos no diretório relativo do storage."""
        target_dir = os.path.join(self.base_path, directory)
        if not os.path.exists(target_dir):
            return []
        return [
            os.path.relpath(os.path.join(dp, f), self.base_path)
            for dp, dn, filenames in os.walk(target_dir)
            for f in filenames
        ]
