"""Definições de dados compartilhadas pelo benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Problem:
    """Um problema isolado da bateria de testes."""

    id: str
    domain: str
    task: str
    example: str


@dataclass(slots=True)
class LLMResponse:
    """Resultado normalizado de uma chamada a um provedor de LLM."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    error: str = ""

    @classmethod
    def failure(cls, error: str) -> "LLMResponse":
        """Cria uma resposta de falha sem consumo de tokens."""
        return cls(
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            success=False,
            error=error,
        )
