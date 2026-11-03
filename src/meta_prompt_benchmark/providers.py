"""Provedores de LLM com interface assíncrona normalizada."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

import google.genai as genai
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential

from .config import API_KEY_VARS, BASE_URLS, BenchmarkConfig
from .models import LLMResponse

logger = logging.getLogger(__name__)

RETRY_WAIT = wait_exponential(multiplier=2, min=5, max=65)
RETRY_STOP = stop_after_attempt(6)

NON_RETRYABLE_ERRORS = (
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "InvalidRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
)

REASONING_MODEL_MARKERS = ("o1", "o3", "o4", "reasoning")


def _is_transient(exc: BaseException) -> bool:
    """Decide se uma exceção vale a pena ser tentada novamente."""
    if any(marker in type(exc).__name__ for marker in NON_RETRYABLE_ERRORS):
        return False

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status", None)

    return not (isinstance(status, int) and 400 <= status < 500 and status not in (408, 429))


def _retry_policy(retry_state: RetryCallState) -> bool:
    if retry_state.outcome is None:
        return True
    exc = retry_state.outcome.exception()
    if exc is None:
        return False
    return _is_transient(exc)


def _build_http_client(ssl_verify: bool, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        verify=ssl_verify,
    )


class LLMProvider(ABC):
    """Classe base de todos os provedores suportados."""

    provider_name: str = "unknown"

    def __init__(
        self,
        model_name: str,
        *,
        ssl_verify: bool = True,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.ssl_verify = ssl_verify
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, prompt: str) -> LLMResponse:
        """Executa o prompt com retry e retorna sempre uma resposta normalizada."""
        try:
            async for attempt in AsyncRetrying(
                wait=RETRY_WAIT,
                stop=RETRY_STOP,
                retry=_retry_policy,
                reraise=True,
            ):
                with attempt:
                    return await self._call_api(prompt)
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            logger.error("Falha definitiva no modelo %s: %s", self.model_name, message)
            return LLMResponse.failure(message)

        return LLMResponse.failure("Sem resposta do provedor")

    @abstractmethod
    async def _call_api(self, prompt: str) -> LLMResponse:
        """Realiza a chamada real à API."""

    async def aclose(self) -> None:  # noqa: B027 - implementação opcional na classe base
        """Libera recursos de rede, quando existirem."""


class _OpenAIChatProvider(LLMProvider):
    """Base para provedores que falam a API de chat da OpenAI."""

    default_base_url: str | None = None

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        base_url = kwargs.pop("base_url", None) or self.default_base_url
        super().__init__(model_name, **kwargs)
        self._http = _build_http_client(self.ssl_verify, self.timeout)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http,
            timeout=self.timeout,
        )

    def _is_reasoning_model(self) -> bool:
        name = self.model_name.lower()
        return any(marker in name for marker in REASONING_MODEL_MARKERS)

    async def _call_api(self, prompt: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self._is_reasoning_model():
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens

        response = await self.client.chat.completions.create(**kwargs)

        if not response.choices:
            return LLMResponse.failure("Provedor retornou sem opções de resposta")

        content = response.choices[0].message.content
        if not content:
            return LLMResponse.failure("Provedor retornou conteúdo vazio")

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        return LLMResponse(content, prompt_tokens, completion_tokens, total_tokens, True)

    async def aclose(self) -> None:
        with suppress(Exception):
            await self._http.aclose()


class OpenAIProvider(_OpenAIChatProvider):
    """Provedor oficial da OpenAI."""

    provider_name = "openai"


class OpenAICompatibleProvider(_OpenAIChatProvider):
    """Provedor genérico para APIs compatíveis com a OpenAI (xAI, DeepSeek, etc.)."""

    def __init__(
        self, provider_name: str, model_name: str, api_key: str, base_url: str, **kwargs: Any
    ) -> None:
        self.provider_name = provider_name
        super().__init__(model_name, api_key, base_url=base_url, **kwargs)


class AnthropicProvider(LLMProvider):
    """Provedor oficial da Anthropic (Claude)."""

    provider_name = "anthropic"

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self._http = _build_http_client(self.ssl_verify, self.timeout)
        self.client = AsyncAnthropic(api_key=api_key, http_client=self._http, timeout=self.timeout)

    async def _call_api(self, prompt: str) -> LLMResponse:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        if not content:
            return LLMResponse.failure("Provedor retornou conteúdo vazio")

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        return LLMResponse(content, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, True)

    async def aclose(self) -> None:
        with suppress(Exception):
            await self._http.aclose()


class GeminiProvider(LLMProvider):
    """Provedor oficial do Google (Gemini) via SDK google-genai."""

    provider_name = "gemini"

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.client = genai.Client(api_key=api_key)

    async def _call_api(self, prompt: str) -> LLMResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        content = response.text
        if not content:
            return LLMResponse.failure("Provedor retornou conteúdo vazio")

        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0

        return LLMResponse(content, prompt_tokens, completion_tokens, total_tokens, True)

    async def aclose(self) -> None:
        aio = getattr(self.client, "aio", None)
        closer = getattr(aio, "aclose", None) or getattr(self.client, "close", None)
        if not callable(closer):
            return
        with suppress(Exception):
            result = closer()
            if inspect.isawaitable(result):
                await result


def create_providers(config: BenchmarkConfig) -> list[LLMProvider]:
    """Instancia os provedores com chave disponível (no dry-run, todos são incluídos)."""
    factories = {
        "openai": lambda key, opts: OpenAIProvider(config.models["openai"], key, **opts),
        "anthropic": lambda key, opts: AnthropicProvider(config.models["anthropic"], key, **opts),
        "gemini": lambda key, opts: GeminiProvider(config.models["gemini"], key, **opts),
        "xai": lambda key, opts: OpenAICompatibleProvider(
            "xai", config.models["xai"], key, BASE_URLS["xai"], **opts
        ),
        "deepseek": lambda key, opts: OpenAICompatibleProvider(
            "deepseek", config.models["deepseek"], key, BASE_URLS["deepseek"], **opts
        ),
    }

    options = {
        "ssl_verify": config.ssl_verify,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    providers: list[LLMProvider] = []
    for name in config.models:
        api_key = config.api_key(name)
        if not api_key:
            if not config.dry_run:
                logger.warning("Chave %s ausente: provedor %s ignorado.", API_KEY_VARS[name], name)
                continue
            api_key = "dry-run-sem-chave"
            logger.info("Dry-run: provedor %s incluído sem chave de API.", name)
        providers.append(factories[name](api_key, options))

    return providers
