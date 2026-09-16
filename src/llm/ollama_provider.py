from langchain_ollama import ChatOllama

from src.llm.base import BaseLLMProvider
from src.utils.config import settings


class OllamaProvider(BaseLLMProvider):
    """
    Provider for local open-source models running through Ollama.
    """

    def get_llm(self):

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )