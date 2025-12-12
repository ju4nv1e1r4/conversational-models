import os
from google import genai
from google.genai import types
from src.utils.config import settings

class LLMService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL

    def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """
        Gera uma resposta simples (não-streamada para este endpoint REST simples).
        """

        if not system_instruction:
            system_instruction = "Você é um assistente útil e direto."

        config = types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=system_instruction)],
            temperature=0.7
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Erro na chamada do LLM: {e}")
            return "Desculpe, tive um problema ao processar sua solicitação."
