from src.llm.ollama_provider import OllamaProvider
from src.utils.config import settings


def get_llm():
    """
    Return the configured local LLM.

    Currently supports Ollama-hosted models.
    """

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaProvider().get_llm()

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )