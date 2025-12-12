from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.llm_service import LLMService
from src.services.redis_service import RedisService
from src.services.intent_service import IntentService

app = FastAPI(title="Compound AI Orchestrator")

llm_service = LLMService()
redis_service = RedisService()
intent_service = IntentService()

POSSIBLE_INTENTS = [
    "saudação", 
    "falar sobre música", 
    "dúvida técnica", 
    "reflexão", 
    "outros"
]

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

@app.post("/chat")
def chat(payload: ChatRequest):
    user_msg = payload.message
    user_id = payload.user_id

    try:
        detected_intent, confidence = intent_service.predict_intent(user_msg, POSSIBLE_INTENTS)
        history = redis_service.get_context_window(user_id)

        system_instruction = "Você é um assistente útil."
        
        if detected_intent == "falar sobre música" and confidence > 0.5:
            system_instruction += " O usuário quer falar sobre música. Fale com ele sobre o assunto"
        elif detected_intent == "reflexão":
            system_instruction += " O usuário parece estar refletindo sobre diversas coisas. Seja um filósofo."

        context_prompt = f"[Sistema: Intenção detectada: {detected_intent} ({confidence:.2f})]\n{user_msg}"

        ai_response = llm_service.generate_response(
            prompt=context_prompt, 
            history=history,
        )
        
        # 5. Persistir
        redis_service.add_message(user_id, "user", user_msg)
        redis_service.add_message(user_id, "model", ai_response)

        return {
            "response": ai_response,
            "metadata": {
                "intent": detected_intent,
                "confidence": confidence,
                "model": llm_service.model_name
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
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
