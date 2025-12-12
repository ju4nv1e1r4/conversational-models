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

    def generate_response(self, prompt: str, history: list = None) -> str:
        """
        Gera resposta considerando o histórico.
        history: Lista de dicts [{'role': 'user', 'content': '...'}, ...]
        """
        
        contents = []

        if history:
            for msg in history:
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text="Você é um assistente útil.")],
            temperature=0.7
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Erro na chamada do LLM: {e}")
            return "Desculpe, tive um problema técnico."
