def build_manual_prompt(task: str) -> str:
    return f"Resolva o seguinte problema:\n{task}"

def build_ai_improved_prompt(task: str) -> str:
    return (
        f"Por favor, atue como um especialista altamente capacitado. "
        f"Analise cuidadosamente o problema abaixo e forneça uma solução completa, "
        f"clara e eficiente, explicando brevemente o seu raciocínio.\n\nProblema:\n{task}"
    )

def build_meta_prompt(task: str, few_shot_example: str) -> str:
    return f"""[Role-Playing]
Você é um Arquiteto de Software Sênior e um Matemático PhD, conhecido por sua lógica impecável e precisão extrema.

[Task Breakdown]
Sua tarefa final é resolver o problema delimitado por ###. Siga rigorosamente estas etapas:
1. Analise o problema e identifique as restrições.
2. Planeje uma abordagem teórica antes de escrever a solução.
3. Implemente o código ou as equações necessárias.
4. Revise meticulosamente seu próprio trabalho.

[Few-Shot Prompting]
Exemplo de formato esperado:
{few_shot_example}

[Chain of Thought]
Não pule direto para a resposta. Explique seu raciocínio passo a passo de forma detalhada.

[Self-Reflection / Critique]
Antes de gerar a saída final, inclua uma "Auto-Crítica". Avalie falhas de segurança, casos extremos ou erros. Após corrigir, apresente a "Resposta Final".

###
Problema:
{task}
###
"""