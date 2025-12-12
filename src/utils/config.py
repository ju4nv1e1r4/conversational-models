import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"

class Settings:
    # Service - LLM
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Infra - Cache (Redis)
    REDIS_CACHE_HOST = os.getenv("REDIS_CACHE_HOST", "redis_cache")
    REDIS_CACHE_PORT = int(os.getenv("REDIS_CACHE_PORT", 6379))
    
    # Infra - Knowledge Graph (FalkorDB on Redis)
    FALKORDB_HOST = os.getenv("FALKORDB_HOST", "falkordb")
    FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))

settings = Settings()
