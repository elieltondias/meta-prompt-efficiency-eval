# Meta-Prompting: Avaliação da Eficiência de Sistemas de IA Generativa na Criação e Otimização Automática de Prompt

Este repositório contém o código-fonte e o framework de testes desenvolvido para o Trabalho de Conclusão de Curso (TCC) focado na análise comparativa de diferentes estratégias de prompting em modelos de linguagem de larga escala (LLMs).

## 📋 Sobre o Projeto

O objetivo deste sistema é mensurar como diferentes abordagens de engenharia de prompt (Manual, AI-Improved e Meta-Prompting) impactam o desempenho, o consumo de tokens e a qualidade das respostas em diversos provedores de IA (OpenAI, Anthropic, Google, DeepSeek, xAI).

### Estratégias Avaliadas
* **Manual**: Prompts diretos e simples.
* **AI-Improved**: Prompts refinados por uma IA para clareza e contexto.
* **Meta-Prompting**: Uso de estruturas complexas de "Role-Playing", "Chain of Thought" e "Self-Reflection" para guiar a IA na resolução de problemas.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Bibliotecas Principais:** * `asyncio` & `httpx`: Para execuções assíncronas e alta concorrência.
    * `pandas`: Manipulação de dados e exportação de resultados.
    * `tenacity`: Implementação de políticas de retry robustas.
    * `python-dotenv`: Gerenciamento de variáveis de ambiente.
* **Modelos Utilizados:** GPT, Claude, Gemini, Grok e DeepSeek seguindo pesquisa do stakoverflow 2025 ([https://survey.stackoverflow.co/2025/](https://survey.stackoverflow.co/2025/))
## 📂 Estrutura do Repositório

* `src/meta-prompting-benchmark/main.py`: Orquestrador assíncrono do benchmark.
* `src/providers.py`: Abstração das APIs de diferentes provedores (OpenAI, Anthropic, Gemini).
* `src/meta-prompting-benchmark/prompts.py`: Motores de construção das três categorias de prompts.
* `src/meta-prompting-benchmark/tests.py`: Bateria de testes contendo problemas de lógica, código e probabilidade.
* `.env.example`: Modelo para configuração das chaves de API.

## 🚀 Como Executar

### 1. Pré-requisitos
Certifique-se de ter o Python instalado e as chaves de API dos provedores que deseja testar.

### 2. Instalação
```bash
# Clone o repositório
git clone [https://github.com/elieltondias/meta-prompt-efficiency-eval.git](https://github.com/elieltondias/meta-prompt-efficiency-eval.git)

# Instale as dependências
pip install -r requirements.txt

# Entre no diretório
cd src/meta-prompting-benchmark/

# Execute

python main.py

