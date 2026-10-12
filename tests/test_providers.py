"""Testes dos provedores de LLM com clientes simulados (sem chamadas de rede)."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from meta_prompt_benchmark.config import API_KEY_VARS, BenchmarkConfig
from meta_prompt_benchmark.models import LLMResponse
from meta_prompt_benchmark.providers import (
    AnthropicProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    _is_transient,
    create_providers,
)


def asyncio_run(coro):
    """Executa uma coroutine de teste sem depender de plugin assíncrono."""
    return asyncio.run(coro)


class FakeProvider(LLMProvider):
    """Provedor de teste controlado."""

    provider_name = "fake"

    def __init__(self, error: Exception | None = None, response: LLMResponse | None = None) -> None:
        super().__init__("fake-model")
        self.error = error
        self.response = response or LLMResponse("ok", 1, 2, 3, True)
        self.calls = 0

    async def _call_api(self, prompt: str) -> LLMResponse:
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class AuthError(Exception):
    """Simula erro de autenticação com status HTTP 401."""

    status_code = 401


class RateLimitError(Exception):
    """Simula erro de limite de requisições com status HTTP 429."""

    status_code = 429


class TransientError(Exception):
    """Simula falha temporária sem status HTTP."""


def test_transient_classification_ignores_client_errors() -> None:
    assert _is_transient(AuthError()) is False
    assert _is_transient(RateLimitError()) is True
    assert _is_transient(TransientError()) is True


def test_transient_classification_rejects_bad_request() -> None:
    class BadRequestError(Exception):
        status_code = 400

    assert _is_transient(BadRequestError()) is False


def test_transient_classification_allows_rate_limit_and_timeout() -> None:
    class TooManyRequests(Exception):
        status_code = 429

    class RequestTimeout(Exception):
        status_code = 408

    assert _is_transient(TooManyRequests()) is True
    assert _is_transient(RequestTimeout()) is True


def test_generate_returns_normalized_success() -> None:
    provider = FakeProvider()

    result = asyncio_run(provider.generate("prompt"))

    assert result.success is True
    assert result.content == "ok"
    assert result.total_tokens == 3


def test_generate_converts_exception_into_failure() -> None:
    provider = FakeProvider(error=AuthError("credenciais inválidas"))

    result = asyncio_run(provider.generate("prompt"))

    assert result.success is False
    assert "credenciais inválidas" in result.error
    assert result.total_tokens == 0


def test_generate_does_not_retry_non_transient_errors() -> None:
    provider = FakeProvider(error=AuthError("401"))

    asyncio_run(provider.generate("prompt"))

    assert provider.calls == 1


class FakeCompletions:
    """Substitui `client.chat.completions.create`."""

    def __init__(self, response) -> None:
        self.response = response
        self.captured_kwargs = None

    async def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return self.response


def make_openai_provider(model_name: str, response) -> tuple[OpenAIProvider, FakeCompletions]:
    provider = OpenAIProvider(model_name, "chave-teste")
    fake = FakeCompletions(response)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return provider, fake


def completion_response(content, usage) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openai_provider_returns_content_and_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    provider, _ = make_openai_provider("gpt-5.5", completion_response("texto", usage))

    result = asyncio_run(provider._call_api("prompt"))

    assert result.success is True
    assert result.content == "texto"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.total_tokens == 18


def test_openai_provider_handles_empty_content() -> None:
    provider, _ = make_openai_provider("gpt-5.5", completion_response(None, None))

    result = asyncio_run(provider._call_api("prompt"))

    assert result.success is False
    assert "vazio" in result.error
    assert result.total_tokens == 0


def test_openai_provider_handles_missing_choices() -> None:
    response = SimpleNamespace(choices=[], usage=None)
    provider, _ = make_openai_provider("gpt-5.5", response)

    result = asyncio_run(provider._call_api("prompt"))

    assert result.success is False


def test_openai_provider_uses_max_tokens_for_standard_models() -> None:
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    provider, fake = make_openai_provider("gpt-5.5", completion_response("ok", usage))

    asyncio_run(provider._call_api("prompt"))

    assert fake.captured_kwargs["temperature"] == 0.7
    assert fake.captured_kwargs["max_tokens"] == 4096
    assert "max_completion_tokens" not in fake.captured_kwargs


def test_openai_provider_uses_completion_budget_for_reasoning_models() -> None:
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    provider, fake = make_openai_provider("o3-mini", completion_response("ok", usage))

    asyncio_run(provider._call_api("prompt"))

    assert fake.captured_kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in fake.captured_kwargs


def test_openai_compatible_provider_configures_base_url_and_name() -> None:
    provider = OpenAICompatibleProvider(
        "deepseek", "deepseek-v4-pro", "chave-teste", "https://api.deepseek.com"
    )

    assert provider.provider_name == "deepseek"
    assert str(provider.client.base_url).rstrip("/") == "https://api.deepseek.com"


def test_anthropic_provider_joins_text_blocks() -> None:
    provider = AnthropicProvider("claude-opus-4-7", "chave-teste")

    block = SimpleNamespace(type="text", text="parte um ")
    other = SimpleNamespace(type="thinking", text="raciocínio")
    response = SimpleNamespace(
        content=[block, other],
        usage=SimpleNamespace(input_tokens=13, output_tokens=9),
    )

    class FakeMessages:
        async def create(self, **kwargs):
            return response

    provider.client = SimpleNamespace(messages=FakeMessages())

    result = asyncio_run(provider._call_api("prompt"))

    assert result.success is True
    assert result.content == "parte um "
    assert result.prompt_tokens == 13
    assert result.completion_tokens == 9
    assert result.total_tokens == 22


def test_anthropic_provider_handles_empty_content() -> None:
    provider = AnthropicProvider("claude-opus-4-7", "chave-teste")
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )

    class FakeMessages:
        async def create(self, **kwargs):
            return response

    provider.client = SimpleNamespace(messages=FakeMessages())

    result = asyncio_run(provider._call_api("prompt"))

    assert result.success is False


def test_create_providers_skips_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "chave-openai")

    config = BenchmarkConfig.load(models=["openai", "anthropic"])
    providers = create_providers(config)

    assert [provider.provider_name for provider in providers] == ["openai"]
    assert providers[0].model_name == "gpt-5.5"


def test_create_providers_applies_benchmark_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "chave-openai")

    config = BenchmarkConfig.load(
        models=["openai"], temperature=0.2, max_tokens=512, timeout=30.0, ssl_verify=False
    )
    provider = create_providers(config)[0]

    assert provider.temperature == 0.2
    assert provider.max_tokens == 512
    assert provider.timeout == 30.0
    assert provider.ssl_verify is False


def test_create_providers_includes_every_provider_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in API_KEY_VARS.values():
        monkeypatch.delenv(key, raising=False)

    providers = create_providers(BenchmarkConfig.load(dry_run=True))

    assert [provider.provider_name for provider in providers] == [
        "openai",
        "anthropic",
        "gemini",
        "xai",
        "deepseek",
    ]
