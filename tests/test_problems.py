"""Testes da bateria de problemas."""

import pytest

from meta_prompt_benchmark.models import Problem
from meta_prompt_benchmark.problems import TEST_BATTERY


def test_battery_has_ten_problems() -> None:
    assert len(TEST_BATTERY) == 10


def test_problem_ids_are_unique() -> None:
    ids = [problem.id for problem in TEST_BATTERY]

    assert len(ids) == len(set(ids))


def test_domains_are_valid() -> None:
    domains = {problem.domain for problem in TEST_BATTERY}

    assert domains == {"codigo", "matematica"}


@pytest.mark.parametrize("problem", TEST_BATTERY, ids=lambda problem: problem.id)
def test_problem_has_all_required_fields(problem: Problem) -> None:
    assert problem.id.strip()
    assert problem.task.strip()
    assert problem.example.strip()
    assert problem.domain in {"codigo", "matematica"}


def test_battery_covers_both_domains() -> None:
    codigo = [p for p in TEST_BATTERY if p.domain == "codigo"]
    matematica = [p for p in TEST_BATTERY if p.domain == "matematica"]

    assert len(codigo) == 5
    assert len(matematica) == 5
