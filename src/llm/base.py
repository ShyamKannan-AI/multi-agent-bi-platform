from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Base interface for supported local LLM providers.
    """

    @abstractmethod
    def get_llm(self) -> Any:
        """
        Return the configured LangChain chat model.
        """
        raise NotImplementedError