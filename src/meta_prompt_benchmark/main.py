"""Orquestrador assíncrono do benchmark de meta-prompting."""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_MODELS, PROMPT_CATEGORIES, BenchmarkConfig, ConfigError
from .models import Problem
from .problems import TEST_BATTERY
from .prompts import build_ai_improved_prompt, build_manual_prompt, build_meta_prompt
from .providers import LLMProvider, create_providers

logger = logging.getLogger(__name__)

PROMPT_BUILDERS = {
    "Manual": lambda problem: build_manual_prompt(problem.task),
    "AI-Improved": lambda problem: build_ai_improved_prompt(problem.task),
    "Meta-Prompt": lambda problem: build_meta_prompt(problem.task, problem.example),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando do benchmark."""
    parser = argparse.ArgumentParser(
        prog="meta-benchmark",
        description=(
            "Compara estratégias de prompt (Manual, AI-Improved, Meta-Prompt) entre provedores de LLM."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="PROVEDOR",
        help=f"Provedores a executar. Disponíveis: {', '.join(sorted(DEFAULT_MODELS))}",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=PROMPT_CATEGORIES,
        metavar="CATEGORIA",
        help="Categorias de prompt a executar.",
    )
    parser.add_argument(
        "--test-ids",
        nargs="+",
        metavar="ID",
        help="IDs de problemas a executar (padrão: toda a bateria).",
    )
    parser.add_argument("--concurrency", type=int, help="Máximo de requisições simultâneas (padrão: 2).")
    parser.add_argument("--temperature", type=float, help="Temperatura das respostas (padrão: 0.7).")
    parser.add_argument("--max-tokens", type=int, help="Limite de tokens de resposta (padrão: 4096).")
    parser.add_argument("--timeout", type=float, help="Timeout de leitura em segundos (padrão: 120).")
    parser.add_argument(
        "--output", metavar="DIR", help="Diretório de saída dos resultados (padrão: results/)."
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Desativa a verificação de certificados TLS (uso restrito a ambientes de teste).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Lista o plano de execução sem chamar nenhuma API."
    )
    parser.add_argument("--verbose", action="store_true", help="Exibe logs de debug.")
    return parser.parse_args(argv)


def select_problems(config: BenchmarkConfig) -> tuple[Problem, ...]:
    """Filtra a bateria de testes conforme os IDs solicitados."""
    if not config.test_ids:
        return TEST_BATTERY

    known = {problem.id for problem in TEST_BATTERY}
    unknown = [test_id for test_id in config.test_ids if test_id not in known]
    if unknown:
        raise ConfigError("IDs de teste desconhecidos: " + ", ".join(unknown))

    return tuple(problem for problem in TEST_BATTERY if problem.id in config.test_ids)


def build_plan(
    providers: list[LLMProvider], problems: tuple[Problem, ...], categories: tuple[str, ...]
) -> list[tuple[LLMProvider, Problem, str, str]]:
    """Gera a lista de execuções (provedor, problema, categoria, prompt)."""
    plan: list[tuple[LLMProvider, Problem, str, str]] = []
    for provider in providers:
        for problem in problems:
            for category in categories:
                plan.append((provider, problem, category, PROMPT_BUILDERS[category](problem)))
    return plan


async def run_single_test(
    provider: LLMProvider,
    problem: Problem,
    category: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Executa uma combinação provedor/problema/categoria e monta a linha do resultado."""
    if config.jitter_max > 0:
        await asyncio.sleep(random.uniform(config.jitter_min, config.jitter_max))

    async with semaphore:
        logger.info("Disparando -> %s | %s | %s", provider.model_name, problem.id, category)

        started_at = time.perf_counter()
        response = await provider.generate(prompt)
        elapsed = time.perf_counter() - started_at

        if response.success:
            logger.info(
                "Concluído (%.2fs) -> %s | %s | %s", elapsed, provider.model_name, problem.id, category
            )
        else:
            logger.warning(
                "Falha -> %s | %s | %s | Erro: %s",
                provider.model_name,
                problem.id,
                category,
                response.error,
            )

        return {
            "Modelo": provider.model_name,
            "Provedor": provider.provider_name,
            "Categoria_Prompt": category,
            "Dominio_Teste": problem.domain,
            "ID_Teste": problem.id,
            "Tempo_Resposta_Segundos": round(elapsed, 2),
            "Tokens_Prompt": response.prompt_tokens,
            "Tokens_Completion": response.completion_tokens,
            "Tokens_Total": response.total_tokens,
            "Tamanho_Resposta_Caracteres": len(response.content),
            "Sucesso": response.success,
            "Mensagem_Erro": response.error,
            "Resposta_IA": response.content if response.success else "ERRO NA GERAÇÃO",
        }


async def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """Executa todas as combinações planejadas e devolve o DataFrame de resultados."""
    problems = select_problems(config)
    providers = create_providers(config)
    if not providers:
        raise ConfigError(
            "Nenhuma chave de API encontrada. Copie .env.example para .env e preencha ao menos uma chave."
        )

    plan = build_plan(providers, problems, config.categories)

    logger.info(
        "%d execuções planejadas (%d provedores x %d problemas x %d categorias), concorrência=%d",
        len(plan),
        len(providers),
        len(problems),
        len(config.categories),
        config.concurrency,
    )

    if config.dry_run:
        for provider, problem, category, _ in plan:
            logger.info("DRY-RUN -> %s | %s | %s", provider.model_name, problem.id, category)
        return pd.DataFrame()

    semaphore = asyncio.Semaphore(config.concurrency)
    try:
        rows = await asyncio.gather(*(run_single_test(*item, semaphore, config) for item in plan))
    finally:
        for provider in providers:
            await provider.aclose()

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega latência, tokens e taxa de sucesso por modelo e categoria."""
    if df.empty:
        return df

    grouped = df.groupby(["Modelo", "Categoria_Prompt"], dropna=False)
    summary = grouped.agg(
        Testes=("ID_Teste", "count"),
        Sucessos=("Sucesso", "sum"),
        Tempo_Medio_Segundos=("Tempo_Resposta_Segundos", "mean"),
        Tokens_Prompt_Medio=("Tokens_Prompt", "mean"),
        Tokens_Completion_Medio=("Tokens_Completion", "mean"),
        Tokens_Total_Medio=("Tokens_Total", "mean"),
    ).reset_index()

    summary["Taxa_Sucesso"] = (summary["Sucessos"] / summary["Testes"]).round(4)
    for column in (
        "Tempo_Medio_Segundos",
        "Tokens_Prompt_Medio",
        "Tokens_Completion_Medio",
        "Tokens_Total_Medio",
    ):
        summary[column] = summary[column].round(2)

    return summary


def export_results(df: pd.DataFrame, config: BenchmarkConfig) -> tuple[Path, Path]:
    """Grava o CSV detalhado e o resumo agregado com carimbo de data/hora."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_path = config.output_dir / f"resultados_benchmark_{stamp}.csv"
    summary_path = config.output_dir / f"resumo_benchmark_{stamp}.csv"

    df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    build_summary(df).to_csv(summary_path, index=False, encoding="utf-8-sig")

    return detailed_path, summary_path


def configure_logging(verbose: bool) -> None:
    """Configura o formato e o nível dos logs da execução."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do comando de linha de comando."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = BenchmarkConfig.load(
            models=args.models,
            categories=args.categories,
            test_ids=args.test_ids,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            output_dir=args.output,
            ssl_verify=False if args.no_ssl_verify else None,
            dry_run=args.dry_run,
        )
    except ConfigError as exc:
        logger.error("Configuração inválida: %s", exc)
        return 2

    if not config.ssl_verify:
        logger.warning("Verificação TLS desativada: as respostas não serão autenticadas.")

    logger.info("Iniciando benchmark de meta-prompting...")

    try:
        dataframe = asyncio.run(run_benchmark(config))
    except ConfigError as exc:
        logger.error(exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        return 130

    if config.dry_run:
        logger.info("Dry-run concluído: nenhuma chamada de API foi realizada.")
        return 0

    detailed_path, summary_path = export_results(dataframe, config)
    logger.info("Resultados gravados em %s", detailed_path)
    logger.info("Resumo agregado gravado em %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
