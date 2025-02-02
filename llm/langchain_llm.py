from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping
import requests

class CustomRESTLLM(LLM):
    api_url: str
    model_kwargs: dict = {}
    
    def __init__(self, api_url: str, **kwargs):
        super().__init__()
        self.api_url = api_url
        self.model_kwargs = kwargs
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Execute LLM call through REST API"""
        response = requests.post(
            self.api_url,
            json={
                "prompt": prompt,
                "stop": stop,
                **self.model_kwargs
            }
        )
        return response.json()["text"]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get parameters used to identify this LLM."""
        return {
            "api_url": self.api_url,
            **self.model_kwargs
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_rest_llm" 