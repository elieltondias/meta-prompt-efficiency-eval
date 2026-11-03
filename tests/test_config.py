"""Testes da configuração do benchmark."""

import pytest

from meta_prompt_benchmark.config import (
    API_KEY_VARS,
    DEFAULT_MODELS,
    MODEL_VARS,
    PROMPT_CATEGORIES,
    BenchmarkConfig,
    ConfigError,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove chaves e parâmetros do ambiente para não vazar entre testes."""
    for key in (
        list(API_KEY_VARS.values())
        + list(MODEL_VARS.values())
        + [
            "BENCHMARK_CONCURRENCY",
            "BENCHMARK_TEMPERATURE",
            "BENCHMARK_MAX_TOKENS",
            "BENCHMARK_TIMEOUT",
            "BENCHMARK_OUTPUT_DIR",
            "DISABLE_SSL_VERIFY",
        ]
    ):
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_sensible() -> None:
    config = BenchmarkConfig.load()

    assert config.models == DEFAULT_MODELS
    assert config.categories == PROMPT_CATEGORIES
    assert config.concurrency == 2
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.ssl_verify is True
    assert config.output_dir.name == "results"
    assert config.enabled_providers() == []


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "8")
    monkeypatch.setenv("BENCHMARK_TEMPERATURE", "0.1")
    monkeypatch.setenv("BENCHMARK_MAX_TOKENS", "1024")

    config = BenchmarkConfig.load()

    assert config.concurrency == 8
    assert config.temperature == 0.1
    assert config.max_tokens == 1024


def test_cli_arguments_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "8")

    config = BenchmarkConfig.load(concurrency=3)

    assert config.concurrency == 3


def test_invalid_integer_in_environment_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_CONCURRENCY", "muitos")

    with pytest.raises(ConfigError, match="BENCHMARK_CONCURRENCY"):
        BenchmarkConfig.load()


def test_unknown_provider_raises() -> None:
    with pytest.raises(ConfigError, match="Provedores desconhecidos"):
        BenchmarkConfig.load(models=["chatgpt"])


def test_unknown_category_raises() -> None:
    with pytest.raises(ConfigError, match="Categorias inválidas"):
        BenchmarkConfig.load(categories=["Chain-of-Thought"])


def test_non_positive_concurrency_raises() -> None:
    with pytest.raises(ConfigError, match="concorrência"):
        BenchmarkConfig.load(concurrency=0)


def test_non_positive_timeout_raises() -> None:
    with pytest.raises(ConfigError, match="timeout"):
        BenchmarkConfig.load(timeout=0)


def test_out_of_range_temperature_raises() -> None:
    with pytest.raises(ConfigError, match="temperatura"):
        BenchmarkConfig.load(temperature=3.0)


def test_non_positive_max_tokens_raises() -> None:
    with pytest.raises(ConfigError, match="tokens"):
        BenchmarkConfig.load(max_tokens=0)


def test_model_filter_keeps_requested_providers_only() -> None:
    config = BenchmarkConfig.load(models=["openai", "gemini"])

    assert set(config.models) == {"openai", "gemini"}


def test_ssl_verification_is_disabled_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_SSL_VERIFY", "true")

    assert BenchmarkConfig.load().ssl_verify is False

    monkeypatch.setenv("DISABLE_SSL_VERIFY", "false")

    assert BenchmarkConfig.load().ssl_verify is True


def test_enabled_providers_require_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "chave-teste")

    config = BenchmarkConfig.load()

    assert config.enabled_providers() == ["openai"]
    assert config.api_key("anthropic") is None


def test_model_names_can_be_overridden_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")

    config = BenchmarkConfig.load(models=["openai"])

    assert config.models["openai"] == "gpt-5-mini"


def test_empty_model_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    with pytest.raises(ConfigError, match="Modelo vazio"):
        BenchmarkConfig.load(models=["openai"])
