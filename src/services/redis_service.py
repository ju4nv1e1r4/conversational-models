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
        self.max_window = 20

    def get_data(self, key: str):
        return self.client.get(key)

    def set_data(self, key: str, data: any):
        return self.client.set(key, data, ex=self.ttl)

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
    
    def clear_history(self, user_id: str) -> bool:
        system_prompt_redis_key = f"llm:prompt:system_prompt:{user_id}"
        context_prompt_redis_key = f"llm:prompt:context_prompt:{user_id}"
        intent_redis_key = f"slm:prompt:intent:{user_id}"
        payloads_redis_key = f"api:payloads:{user_id}"

        self.client.delete(f"session:{user_id}")
        self.client.delete(system_prompt_redis_key)
        self.client.delete(context_prompt_redis_key)
        self.client.delete(intent_redis_key)
        self.client.delete(payloads_redis_key)

        return True
