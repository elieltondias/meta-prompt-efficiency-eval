# Contribuindo

Obrigado por ajudar a evoluir o benchmark de meta-prompting. Este documento descreve o fluxo de trabalho usado no projeto.

## Requisitos

- Python 3.11 ou superior
- `pip` e `venv`

## Preparando o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

## Executando localmente

```bash
cp .env.example .env
# preencha ao menos uma chave de API no .env

PYTHONPATH=src python -m meta_prompt_benchmark --dry-run
PYTHONPATH=src python -m meta_prompt_benchmark
```

## Qualidade de código

Antes de abrir um pull request, execute:

```bash
ruff check src tests
ruff format --check src tests
pytest
```

O mesmo conjunto de verificações roda automaticamente no CI (GitHub Actions) e, se o `pre-commit` estiver instalado, a cada commit.

## Estrutura das mudanças

| Área | Arquivo |
|---|---|
| Orquestrador e CLI | `src/meta_prompt_benchmark/main.py` |
| Configuração | `src/meta_prompt_benchmark/config.py` |
| Provedores de LLM | `src/meta_prompt_benchmark/providers.py` |
| Construtores de prompt | `src/meta_prompt_benchmark/prompts.py` |
| Bateria de testes do benchmark | `src/meta_prompt_benchmark/problems.py` |
| Testes automatizados | `tests/` |

## Diretrizes

- Sem comentários `#` inline: o código deve se explicar por nomes e docstrings.
- Documentação, docstrings e mensagens de log em português do Brasil.
- Novos provedores devem herdar de `LLMProvider` e retornar `LLMResponse`.
- Toda funcionalidade nova precisa de teste em `tests/`.
- Nunca commite chaves de API, o arquivo `.env` ou os CSVs de `results/`.

## Commits

Use mensagens curtas e descritivas no imperativo, por exemplo:

- `Adiciona filtro por domínio na bateria de testes`
- `Corrige tratamento de resposta vazia no provedor OpenAI`
- `Adiciona resumo agregado ao exportar resultados`
