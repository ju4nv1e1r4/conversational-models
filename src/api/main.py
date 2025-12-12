from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.llm_service import LLMService
# TODO: from src.services.redis_service import RedisService (next step)

app = FastAPI(title="Compound AI Orchestrator")
llm_service = LLMService()

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

@app.get("/health")
def health():
    return {"status": "running", "model": llm_service.model_name}

@app.post("/chat")
def chat(payload: ChatRequest):
    # TODO: Check redis cace
    # TODO: Enrich with graphrag (FalkorDB)
    # TODO: Classify intention (KServe/ONNX)

    try:
        response_text = llm_service.generate_response(payload.message)
        
        return {
            "response": response_text,
            "metadata": {
                "source": "llm_direct", # It will be changed for cache or rag soon
                "model": llm_service.model_name
            }
        }
    except Exception as request_error:
        raise HTTPException(status_code=500, detail=str(request_error))
    except Exception as undefined_error:
        return {
            "response": "Desculpe, tive um problema ao processar sua solicitação." or response_text,
            "metadata": {
                "source": "llm_direct",
                "model": llm_service.model_name,
                "error": str(undefined_error)
            }
        }