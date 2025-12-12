from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.llm_service import LLMService
from src.services.redis_service import RedisService

app = FastAPI(title="Compound AI Orchestrator")

llm_service = LLMService()
redis_service = RedisService()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user" # NOTE: to separate sessions

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat(payload: ChatRequest):
    user_msg = payload.message
    user_id = payload.user_id

    try:
        history = redis_service.get_context_window(user_id)

        ai_response = llm_service.generate_response(prompt=user_msg, history=history)

        redis_service.add_message(user_id, "user", user_msg)
        redis_service.add_message(user_id, "model", ai_response)

        return {
            "response": ai_response,
            "metadata": {
                "history_len": len(history),
                "user_id": user_id
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as undefined_error:
        return {
            "response": "Desculpe, tive um problema ao processar sua solicitação." or ai_response,
            "metadata": {
                "source": "llm_direct",
                "model": llm_service.model_name,
                "error": str(undefined_error)
            }
        }
