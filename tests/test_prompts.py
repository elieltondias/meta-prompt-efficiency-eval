"""Testes dos construtores de prompt."""

from meta_prompt_benchmark.prompts import (
    build_ai_improved_prompt,
    build_manual_prompt,
    build_meta_prompt,
)

TASK = "Resolva o problema X."
EXAMPLE = "- Planejamento: passo um.\n- Execução: passo dois."


def test_manual_prompt_contains_task() -> None:
    prompt = build_manual_prompt(TASK)

    assert TASK in prompt
    assert len(prompt) < len(build_ai_improved_prompt(TASK))


def test_ai_improved_prompt_contains_task_and_role() -> None:
    prompt = build_ai_improved_prompt(TASK)

    assert TASK in prompt
    assert "especialista" in prompt
    assert "raciocínio" in prompt


def test_meta_prompt_contains_task_and_example() -> None:
    prompt = build_meta_prompt(TASK, EXAMPLE)

    assert TASK in prompt
    assert EXAMPLE in prompt


def test_meta_prompt_contains_all_strategy_sections() -> None:
    prompt = build_meta_prompt(TASK, EXAMPLE)

    for section in (
        "Role-Playing",
        "Task Breakdown",
        "Few-Shot Prompting",
        "Chain of Thought",
        "Self-Reflection",
    ):
        assert section in prompt


def test_meta_prompt_is_the_longest_strategy() -> None:
    lengths = {
        len(build_manual_prompt(TASK)),
        len(build_ai_improved_prompt(TASK)),
        len(build_meta_prompt(TASK, EXAMPLE)),
    }

    assert len(lengths) == 3
    assert max(lengths) == len(build_meta_prompt(TASK, EXAMPLE))
