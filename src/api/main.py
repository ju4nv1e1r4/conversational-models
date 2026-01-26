from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json

from src.services.llm_service import LLMService
from src.services.redis_service import RedisService
from src.services.intent_service import IntentService
from src.utils.telemetry import telemetry

@asynccontextmanager
async def on_startup(app: FastAPI):
    redis_service.clear_history("default_user")
    yield

app = FastAPI(title="Compound AI Orchestrator", lifespan=on_startup)
telemetry.instrument_app(app)

llm_service = LLMService()
redis_service = RedisService()
intent_service = IntentService()

POSSIBLE_INTENTS = [
    "saudação",
    "dúvida técnica",
    "aprendizado",
    "reflexão",
    "despedida",
]

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

@app.post("/chat")
def chat(payload: ChatRequest):
    user_msg = payload.message
    user_id = payload.user_id

    system_prompt_redis_key = f"llm:prompt:system_prompt:{user_id}"
    context_prompt_redis_key = f"llm:prompt:context_prompt:{user_id}"
    intent_redis_key = f"slm:prompt:intent:{user_id}"
    payloads_redis_key = f"api:payloads:{user_id}"

    try:
        detected_intent, confidence = intent_service.predict_intent(user_msg, POSSIBLE_INTENTS)
        redis_service.set_data(intent_redis_key, detected_intent)

        history = redis_service.get_context_window(user_id)

        system_instruction = "Você é um assistente útil. Você é capaz de ajudar o usuário em diversos assuntos."

        if detected_intent == "saudação" and confidence > 0.7:
            system_instruction += " O usuário quer falar sobre música. Fale com ele sobre o assunto"
            redis_service.set_data(system_prompt_redis_key, system_instruction)
        elif detected_intent == "reflexão" and confidence > 0.7:
            system_instruction += " O usuário parece estar refletindo sobre diversas coisas. Ajude-o a refletir."
            redis_service.set_data(system_prompt_redis_key, system_instruction)
        elif detected_intent == "dúvida técnica" and confidence > 0.7:
            system_instruction += " O usuário parece ter alguma dúvida técnica. Seja receptivo e ajude-o como for necessário"
            redis_service.set_data(system_prompt_redis_key, system_instruction)
        elif detected_intent == "aprendizado" and confidence > 0.7:
            system_instruction += " O usuário parece que quer que você o ensine algo. Seja um professor."
            redis_service.set_data(system_prompt_redis_key, system_instruction)
        elif detected_intent == "despedida" and confidence > 0.7:
            system_instruction += " O usuário parece estar se despedindo. Finalize a conversa."
            redis_service.set_data(system_prompt_redis_key, system_instruction)

        context_prompt = f"[Intenção detectada: {detected_intent}; Confiança: {confidence:.2f}]\nMensagem do usuário: {user_msg}"
        redis_service.set_data(context_prompt_redis_key, context_prompt)

        ai_response = llm_service.generate_response(
            prompt=context_prompt, 
            history=history,
        )

        # persist msgs on redis (short term memory)
        redis_service.add_message(user_id, "user", user_msg)
        redis_service.add_message(user_id, "model", ai_response)

        final_payload = {
            "response": ai_response,
            "metadata": {
                "intent": detected_intent,
                "confidence": confidence,
                "model": llm_service.model_name
            }
        }
        redis_service.set_data(payloads_redis_key, json.dumps(final_payload, ensure_ascii=False))

        return final_payload

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as undefined_error:
        error_response = {
            "response": "Desculpe, tive um problema ao processar sua solicitação." or ai_response,
            "metadata": {
                "source": "llm_direct",
                "model": llm_service.model_name,
                "error": str(undefined_error)
            }
        }
        redis_service.set_data(payloads_redis_key, json.dumps(error_response, ensure_ascii=False))
        return error_response
