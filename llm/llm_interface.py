import requests
from typing import Dict, Any

class LLMInterface:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    async def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                **kwargs
            }
        )
        return response.json()["text"] 