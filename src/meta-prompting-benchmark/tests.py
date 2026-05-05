TEST_BATTERY = [
    # ==========================================
    # PROBLEMAS DE CÓDIGO
    # ==========================================
    {
        "id": "CODE_01_SORTING",
        "task": "Escreva um algoritmo em Python para realizar um Merge Sort em uma lista de dicionários, ordenando de forma decrescente com base em uma chave específica 'idade'.",
        "example": "- Planejamento: Divisão recursiva.\n- Execução: Implementar a função intercalando os resultados inversamente considerando a chave."
    },
    {
        "id": "CODE_02_LRU_CACHE",
        "task": "Implemente uma estrutura de dados LRU (Least Recently Used) Cache em Python orientada a objetos com capacidade O(1) para as operações `get` e `put`. Não utilize functools.lru_cache. Implemente do zero usando Dicionário e Lista Duplamente Encadeada.",
        "example": "- Planejamento: Usar um hash map apontando para nós de uma doubly linked list.\n- Execução: Criar a classe Node e atualizar a cabeça/cauda nas operações."
    },
    {
        "id": "CODE_03_GRAPH_DIJKSTRA",
        "task": "Implemente o algoritmo de Dijkstra em Python para encontrar o caminho mais curto entre um nó de origem e todos os outros nós em um grafo direcionado e ponderado representado por uma lista de adjacências. Retorne também o caminho reconstruído para um nó alvo específico.",
        "example": "- Planejamento: Usar uma fila de prioridade (heapq) para extrair o menor custo atual.\n- Execução: Atualizar o dicionário de distâncias e manter um dicionário de predecessores."
    },
    {
        "id": "CODE_04_CONCURRENCY",
        "task": "Escreva um script em Python utilizando `asyncio` e `aiohttp` para fazer o download simultâneo de 50 URLs diferentes. Implemente um limite de concorrência usando Semaphore para não exceder 10 requisições simultâneas e inclua tratamento de erros.",
        "example": "- Planejamento: Criar o worker para o fetch usando aiohttp.ClientSession e envolver com o asyncio.Semaphore.\n- Execução: Usar asyncio.gather para agrupar as tarefas."
    },
    {
        "id": "CODE_05_DP_KNAPSACK",
        "task": "Resolva o problema clássico da Mochila 0-1 (Knapsack) em Python usando Programação Dinâmica. A função deve receber os arrays `valores`, `pesos` e a capacidade máxima `W`, e retornar o valor máximo que pode ser obtido.",
        "example": "- Planejamento: Matriz DP onde dp[i][w] é o valor máximo com 'i' itens na capacidade 'w'.\n- Execução: Construir a tabela de baixo para cima considerando se inclui ou não o item atual."
    },

    # ==========================================
    # PROBLEMAS DE PROBABILIDADE
    # ==========================================
    {
        "id": "MATH_01_HYPERGEOMETRIC",
        "task": "Uma urna tem 5 bolas vermelhas e 7 azuis. Retiram-se 3 bolas sem reposição. Qual a probabilidade de exatamente 2 serem vermelhas?",
        "example": "- Planejamento: Uso da distribuição hipergeométrica.\n- Execução: (C(5,2) * C(7,1)) / C(12,3)."
    },
    {
        "id": "MATH_02_BAYES",
        "task": "Uma doença afeta 1% de uma população. Um teste para essa doença é 95% preciso para resultados positivos (verdadeiro positivo) e 90% preciso para resultados negativos (verdadeiro negativo). Se uma pessoa é escolhida ao acaso e testa positivo, qual a probabilidade de ela realmente ter a doença?",
        "example": "- Planejamento: Aplicar o Teorema de Bayes.\n- Execução: P(Doença|Positivo) = (P(Positivo|Doença)*P(Doença)) / P(Positivo)."
    },
    {
        "id": "MATH_03_POISSON",
        "task": "Um servidor de banco de dados recebe, em média, 4 requisições de timeout por hora. Assumindo que essas ocorrências seguem uma distribuição de Poisson, qual é a probabilidade de ocorrerem exatamente 6 timeouts em um intervalo de duas horas?",
        "example": "- Planejamento: Ajustar a taxa (lambda) para o novo intervalo e aplicar a fórmula de Poisson.\n- Execução: Novo lambda = 8. P(X=6) = (e^(-8) * 8^6) / 6!."
    },
    {
        "id": "MATH_04_EXPECTED_VALUE",
        "task": "Você lança dois dados de 6 faces justos. Se a soma dos dados for um número primo, você ganha o valor da soma em reais. Se não for primo, você perde 5 reais. Qual é o valor esperado dessa aposta?",
        "example": "- Planejamento: Listar as somas possíveis, identificar os primos, calcular a probabilidade de cada um e o payoff.\n- Execução: Somatório de (P(soma) * payoff) para todas as somas de 2 a 12."
    },
    {
        "id": "MATH_05_COMBINATORICS",
        "task": "Qual é a probabilidade de, ao retirar 5 cartas aleatoriamente de um baralho padrão de 52 cartas sem reposição, obter um 'Full House' (uma trinca e um par)?",
        "example": "- Planejamento: Contar formas de escolher os valores da trinca e do par, e os naipes de cada um.\n- Execução: Casos favoráveis = C(13,1)*C(4,3) * C(12,1)*C(4,2). Casos totais = C(52,5)."
    }
]