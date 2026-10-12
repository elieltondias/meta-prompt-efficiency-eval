"""Testes do orquestrador do benchmark."""

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from meta_prompt_benchmark.config import API_KEY_VARS, BenchmarkConfig, ConfigError
from meta_prompt_benchmark.main import (
    build_plan,
    build_summary,
    export_results,
    main,
    run_single_test,
    select_problems,
)
from meta_prompt_benchmark.models import LLMResponse, Problem
from meta_prompt_benchmark.problems import TEST_BATTERY
from meta_prompt_benchmark.providers import LLMProvider

PROBLEM = TEST_BATTERY[0]


class FakeProvider(LLMProvider):
    """Provedor de teste que devolve uma resposta pré-definida."""

    provider_name = "fake"

    def __init__(self, response: LLMResponse, model_name: str = "fake-model") -> None:
        super().__init__(model_name)
        self.response = response
        self.prompts: list[str] = []

    async def _call_api(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return self.response


def success_response() -> LLMResponse:
    return LLMResponse("resposta", 10, 5, 15, True)


def make_config(**kwargs) -> BenchmarkConfig:
    kwargs.setdefault("jitter_min", 0)
    kwargs.setdefault("jitter_max", 0)
    return BenchmarkConfig(**kwargs)


def test_select_problems_returns_full_battery_by_default() -> None:
    config = make_config()

    assert select_problems(config) == TEST_BATTERY


def test_select_problems_filters_by_id() -> None:
    config = make_config(test_ids=("MATH_02_BAYES",))

    assert [problem.id for problem in select_problems(config)] == ["MATH_02_BAYES"]


def test_select_problems_rejects_unknown_id() -> None:
    config = make_config(test_ids=("NAO_EXISTE",))

    with pytest.raises(ConfigError, match="desconhecidos"):
        select_problems(config)


def test_build_plan_multiplies_providers_problems_and_categories() -> None:
    provider = FakeProvider(success_response())
    problems = TEST_BATTERY[:2]

    plan = build_plan([provider], problems, ("Manual", "Meta-Prompt"))

    assert len(plan) == 1 * 2 * 2
    assert all(prompt for _, _, _, prompt in plan)


def test_build_plan_uses_requested_prompts() -> None:
    provider = FakeProvider(success_response())

    plan = build_plan([provider], (PROBLEM,), ("Meta-Prompt",))

    _, _, _, prompt = plan[0]
    assert PROBLEM.task in prompt
    assert PROBLEM.example in prompt


def test_run_single_test_returns_complete_row() -> None:
    provider = FakeProvider(success_response())
    config = make_config()
    semaphore = asyncio.Semaphore(1)

    row = asyncio.run(run_single_test(provider, PROBLEM, "Manual", "prompt", semaphore, config))

    assert row["Modelo"] == "fake-model"
    assert row["Provedor"] == "fake"
    assert row["Categoria_Prompt"] == "Manual"
    assert row["Dominio_Teste"] == PROBLEM.domain
    assert row["ID_Teste"] == PROBLEM.id
    assert row["Sucesso"] is True
    assert row["Tokens_Total"] == 15
    assert row["Resposta_IA"] == "resposta"
    assert row["Mensagem_Erro"] == ""
    assert row["Tempo_Resposta_Segundos"] >= 0


def test_run_single_test_records_failure() -> None:
    provider = FakeProvider(LLMResponse.failure("chave inválida"))
    config = make_config()
    semaphore = asyncio.Semaphore(1)

    row = asyncio.run(run_single_test(provider, PROBLEM, "Manual", "prompt", semaphore, config))

    assert row["Sucesso"] is False
    assert row["Mensagem_Erro"] == "chave inválida"
    assert row["Resposta_IA"] == "ERRO NA GERAÇÃO"
    assert row["Tamanho_Resposta_Caracteres"] == 0


def test_build_summary_aggregates_by_model_and_category() -> None:
    frame = pd.DataFrame(
        [
            {
                "Modelo": "modelo-a",
                "Categoria_Prompt": "Manual",
                "ID_Teste": "T1",
                "Tempo_Resposta_Segundos": 1.0,
                "Tokens_Prompt": 10,
                "Tokens_Completion": 5,
                "Tokens_Total": 15,
                "Sucesso": True,
            },
            {
                "Modelo": "modelo-a",
                "Categoria_Prompt": "Manual",
                "ID_Teste": "T2",
                "Tempo_Resposta_Segundos": 3.0,
                "Tokens_Prompt": 20,
                "Tokens_Completion": 10,
                "Tokens_Total": 30,
                "Sucesso": False,
            },
        ]
    )

    summary = build_summary(frame)

    assert len(summary) == 1
    assert summary.loc[0, "Testes"] == 2
    assert summary.loc[0, "Sucessos"] == 1
    assert summary.loc[0, "Taxa_Sucesso"] == 0.5
    assert summary.loc[0, "Tempo_Medio_Segundos"] == 2.0
    assert summary.loc[0, "Tokens_Total_Medio"] == 22.5


def test_build_summary_keeps_empty_frame_untouched() -> None:
    empty = pd.DataFrame()

    assert build_summary(empty).empty


def test_export_results_writes_detailed_and_summary_files(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "Modelo": "modelo-a",
                "Categoria_Prompt": "Manual",
                "ID_Teste": "T1",
                "Tempo_Resposta_Segundos": 1.0,
                "Tokens_Prompt": 10,
                "Tokens_Completion": 5,
                "Tokens_Total": 15,
                "Sucesso": True,
            }
        ]
    )
    config = make_config(output_dir=tmp_path)

    detailed_path, summary_path = export_results(frame, config)

    assert detailed_path.exists()
    assert summary_path.exists()
    assert detailed_path.parent == tmp_path
    assert detailed_path.name.startswith("resultados_benchmark_")
    assert summary_path.name.startswith("resumo_benchmark_")

    detailed = pd.read_csv(detailed_path)
    assert set(detailed["Modelo"]) == {"modelo-a"}


def test_build_plan_accepts_multiple_problems() -> None:
    provider = FakeProvider(success_response())
    problems: tuple[Problem, ...] = TEST_BATTERY[:3]

    plan = build_plan([provider], problems, ("Manual", "AI-Improved", "Meta-Prompt"))

    assert len(plan) == 9


def test_main_returns_zero_on_dry_run() -> None:
    code = main(["--dry-run", "--models", "openai", "--test-ids", "MATH_01_HYPERGEOMETRIC"])

    assert code == 0


def test_main_returns_two_on_invalid_configuration() -> None:
    assert main(["--models", "provedor-inexistente"]) == 2


def test_main_returns_two_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in API_KEY_VARS.values():
        monkeypatch.delenv(key, raising=False)

    assert main([]) == 2
