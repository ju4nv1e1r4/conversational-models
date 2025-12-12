import redis
import json
from src.utils.config import settings

class RedisService:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_CACHE_HOST,
            port=settings.REDIS_CACHE_PORT,
            decode_responses=True
        )
        self.ttl = 3600 # 1 hour on session
        self.max_window = 10 # NOTE: context window size

    def add_message(self, user_id: str, role: str, content: str):
        """
        Salva uma mensagem na lista.
        Role deve ser 'user' ou 'model' (padrão Gemini).
        """
        key = f"session:{user_id}"
        message = json.dumps({"role": role, "content": content})
        
        self.client.rpush(key, message)
        self.client.expire(key, self.ttl)

    def get_context_window(self, user_id: str):
        """
        Recupera as últimas N mensagens para enviar ao LLM.
        """
        key = f"session:{user_id}"
        messages_json = self.client.lrange(key, -self.max_window, -1)
        
        return [json.loads(m) for m in messages_json]
    
    def clear_history(self, user_id: str):
        self.client.delete(f"session:{user_id}")
