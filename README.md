# Meta-Prompting: Avaliação da Eficiência de Sistemas de IA Generativa na Criação e Otimização Automática de Prompt

[![CI](https://github.com/elieltondias/meta-prompt-efficiency-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/elieltondias/meta-prompt-efficiency-eval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Repositório do Trabalho de Conclusão de Curso (TCC) com o framework de benchmark comparativo de estratégias de prompting em modelos de linguagem de larga escala (LLMs).

## 📋 Sobre o Projeto

O sistema mensura como diferentes abordagens de engenharia de prompt impactam **latência**, **consumo de tokens** e **taxa de sucesso** em diversos provedores de IA (OpenAI, Anthropic, Google, DeepSeek e xAI).

### Estratégias avaliadas

| Categoria | Descrição |
|---|---|
| **Manual** | Prompt direto, sem instruções adicionais. |
| **AI-Improved** | Prompt refinado com papel, contexto e pedido explícito de raciocínio. |
| **Meta-Prompt** | Estrutura completa com *Role-Playing*, *Task Breakdown*, *Few-Shot*, *Chain of Thought* e *Self-Reflection*. |

### Bateria de testes

10 problemas igualmente distribuídos entre dois domínios:

- **Código (5):** Merge Sort, LRU Cache, Dijkstra, concorrência assíncrona e Mochila 0-1.
- **Matemática (5):** Hipergeométrica, Teorema de Bayes, Poisson, valor esperado e combinatoria.

O modelo de seleção segue a pesquisa Stack Overflow 2025 ([survey.stackoverflow.co/2025](https://survey.stackoverflow.co/2025/)).

## 🛠️ Tecnologias

- **Linguagem:** Python 3.11+
- **Assíncrono:** `asyncio` + `httpx` com limite de concorrência controlado por `Semaphore`
- **Dados:** `pandas` para exportação e agregação de resultados
- **Confiabilidade:** `tenacity` com backoff exponencial (5s → 65s, até 6 tentativas)
- **Configuração:** `python-dotenv`
- **Qualidade:** `ruff` (lint + formatação), `pytest` e `pre-commit`

## 📂 Estrutura do repositório

```
.
├── src/meta_prompt_benchmark/
│   ├── main.py            # Orquestrador assíncrono e CLI
│   ├── config.py          # Configuração (ambiente + linha de comando)
│   ├── providers.py       # Abstração das APIs (OpenAI, Anthropic, Gemini, xAI, DeepSeek)
│   ├── prompts.py         # Construtores das três categorias de prompt
│   ├── problems.py        # Bateria de testes (código e matemática)
│   └── models.py          # Estruturas de dados compartilhadas
├── tests/                 # Testes automatizados (pytest)
├── results/               # CSVs gerados (não versionados)
├── .github/workflows/     # Pipeline de CI
├── .env.example           # Modelo de configuração das chaves de API
└── pyproject.toml         # Configuração de ruff e pytest
```

## 🚀 Como executar

### 1. Pré-requisitos

- Python 3.11 ou superior
- Chaves de API dos provedores que deseja testar (basta uma)

### 2. Instalação

```bash
git clone https://github.com/elieltondias/meta-prompt-efficiency-eval.git
cd meta-prompt-efficiency-eval

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edite .env e preencha ao menos uma chave de API
```

### 3. Execução

```bash
# verifica o plano de execução sem gastar tokens
PYTHONPATH=src python -m meta_prompt_benchmark --dry-run

# executa a bateria completa
PYTHONPATH=src python -m meta_prompt_benchmark
```

### 4. Opções de linha de comando

```bash
PYTHONPATH=src python -m meta_prompt_benchmark --help
```

| Opção | Descrição | Padrão |
|---|---|---|
| `--models` | Provedores a executar (`openai anthropic gemini xai deepseek`) | todos com chave |
| `--categories` | Categorias (`Manual`, `AI-Improved`, `Meta-Prompt`) | todas |
| `--test-ids` | IDs específicos da bateria (ex.: `MATH_02_BAYES`) | bateria completa |
| `--concurrency` | Requisições simultâneas | `2` |
| `--temperature` | Temperatura das respostas | `0.7` |
| `--max-tokens` | Limite de tokens de resposta | `4096` |
| `--timeout` | Timeout de leitura (s) | `120` |
| `--output` | Diretório de saída | `results/` |
| `--no-ssl-verify` | Desativa a verificação TLS (apenas teste) | desativado |
| `--dry-run` | Lista as execuções sem chamar nenhuma API | desativado |
| `--verbose` | Logs de debug | desativado |

Exemplo com filtros:

```bash
PYTHONPATH=src python -m meta_prompt_benchmark \
  --models openai anthropic \
  --categories Manual "Meta-Prompt" \
  --concurrency 4 \
  --output results/experimento-01
```

## ⚙️ Configuração por variáveis de ambiente

Todas as opções também podem ser definidas no `.env`:

| Variável | Descrição | Padrão |
|---|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY` | Chaves de API (ao menos uma obrigatória) | — |
| `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `XAI_MODEL`, `DEEPSEEK_MODEL` | Modelo usado por provedor | `gpt-5.5`, `claude-opus-4-7`, `gemini-3.1-pro-preview`, `grok-4.3`, `deepseek-v4-pro` |
| `BENCHMARK_CONCURRENCY` | Requisições simultâneas | `2` |
| `BENCHMARK_TEMPERATURE` | Temperatura das respostas | `0.7` |
| `BENCHMARK_MAX_TOKENS` | Limite de tokens de resposta | `4096` |
| `BENCHMARK_TIMEOUT` | Timeout de leitura em segundos | `120` |
| `BENCHMARK_OUTPUT_DIR` | Diretório de saída | `results` |
| `DISABLE_SSL_VERIFY` | `true` desativa a verificação de certificados TLS | `false` |

> ⚠️ Os provedores OpenAI, Anthropic, xAI e DeepSeek respeitam `DISABLE_SSL_VERIFY`. O SDK do Gemini mantém a verificação TLS sempre ativa.

## 📊 Resultados

Cada execução gera dois arquivos carimbados por data/hora em `results/`:

- **`resultados_benchmark_AAAAMMDD_HHMMSS.csv`** — linha por execução, com modelo, categoria, domínio, ID do teste, latência, tokens (prompt/completion/total), sucesso, mensagem de erro e a resposta completa da IA.
- **`resumo_benchmark_AAAAMMDD_HHMMSS.csv`** — agregação por modelo × categoria com testes, sucessos, taxa de sucesso, latência média e tokens médios.

Os CSVs e o diretório `results/` não são versionados.

## 🧪 Desenvolvimento

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install

ruff check src tests        # lint
ruff format src tests       # formatação
pytest                      # testes
```

O workflow de CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) executa formatação, lint e testes em Python 3.11 e 3.12 a cada push e pull request.

Veja mais detalhes em [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 Licença


