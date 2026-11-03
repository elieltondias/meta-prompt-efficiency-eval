"""Configuração do benchmark carregada do ambiente e da linha de comando."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.5",
    "anthropic": "claude-opus-4-7",
    "gemini": "gemini-3.1-pro-preview",
    "xai": "grok-4.3",
    "deepseek": "deepseek-v4-pro",
}

API_KEY_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

MODEL_VARS: dict[str, str] = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "xai": "XAI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
}

BASE_URLS: dict[str, str] = {
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
}

PROMPT_CATEGORIES: tuple[str, ...] = ("Manual", "AI-Improved", "Meta-Prompt")


class ConfigError(ValueError):
    """Erro de configuração detectado antes de qualquer chamada de rede."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} deve ser um inteiro, recebido: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} deve ser um número, recebido: {raw!r}") from exc


@dataclass(slots=True)
class BenchmarkConfig:
    """Parâmetros de execução do benchmark."""

    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    categories: tuple[str, ...] = PROMPT_CATEGORIES
    test_ids: tuple[str, ...] = ()
    concurrency: int = 2
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0
    jitter_min: float = 0.1
    jitter_max: float = 2.0
    ssl_verify: bool = True
    output_dir: Path = field(default_factory=lambda: Path("results"))
    dry_run: bool = False

    @classmethod
    def load(
        cls,
        *,
        models: list[str] | None = None,
        categories: list[str] | None = None,
        test_ids: list[str] | None = None,
        concurrency: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        output_dir: str | None = None,
        ssl_verify: bool | None = None,
        dry_run: bool = False,
    ) -> BenchmarkConfig:
        """Combina variáveis de ambiente com os argumentos da linha de comando."""
        load_dotenv()

        resolved_models = {
            name: (os.getenv(MODEL_VARS[name]) or default).strip() for name, default in DEFAULT_MODELS.items()
        }
        if models:
            unknown = [m for m in models if m not in resolved_models]
            if unknown:
                raise ConfigError(
                    "Provedores desconhecidos: "
                    + ", ".join(unknown)
                    + ". Disponíveis: "
                    + ", ".join(sorted(resolved_models))
                )
            resolved_models = {name: resolved_models[name] for name in models}

        empty_models = [name for name, value in resolved_models.items() if not value]
        if empty_models:
            raise ConfigError("Modelo vazio configurado para: " + ", ".join(empty_models))

        resolved_categories = tuple(categories) if categories else PROMPT_CATEGORIES
        unknown_categories = [c for c in resolved_categories if c not in PROMPT_CATEGORIES]
        if unknown_categories:
            raise ConfigError(
                "Categorias inválidas: "
                + ", ".join(unknown_categories)
                + ". Disponíveis: "
                + ", ".join(PROMPT_CATEGORIES)
            )

        resolved_concurrency = (
            concurrency if concurrency is not None else _env_int("BENCHMARK_CONCURRENCY", 2)
        )
        if resolved_concurrency < 1:
            raise ConfigError("A concorrência deve ser maior que zero.")

        resolved_timeout = timeout if timeout is not None else _env_float("BENCHMARK_TIMEOUT", 120.0)
        if resolved_timeout <= 0:
            raise ConfigError("O timeout deve ser maior que zero.")

        resolved_temperature = (
            temperature if temperature is not None else _env_float("BENCHMARK_TEMPERATURE", 0.7)
        )
        if not 0 <= resolved_temperature <= 2:
            raise ConfigError("A temperatura deve estar entre 0 e 2.")

        resolved_max_tokens = max_tokens if max_tokens is not None else _env_int("BENCHMARK_MAX_TOKENS", 4096)
        if resolved_max_tokens < 1:
            raise ConfigError("O limite de tokens deve ser maior que zero.")

        resolved_ssl = ssl_verify if ssl_verify is not None else not _env_bool("DISABLE_SSL_VERIFY", False)

        resolved_output = output_dir or os.getenv("BENCHMARK_OUTPUT_DIR") or "results"

        return cls(
            models=resolved_models,
            categories=resolved_categories,
            test_ids=tuple(test_ids) if test_ids else (),
            concurrency=resolved_concurrency,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            timeout=resolved_timeout,
            ssl_verify=resolved_ssl,
            output_dir=Path(resolved_output),
            dry_run=dry_run,
        )

    def api_key(self, provider: str) -> str | None:
        """Retorna a chave de API do provedor, se estiver definida."""
        return os.getenv(API_KEY_VARS[provider]) or None

    def enabled_providers(self) -> list[str]:
        """Lista os provedores com chave de API disponível."""
        return [name for name in self.models if self.api_key(name)]
