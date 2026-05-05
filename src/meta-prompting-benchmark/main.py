import os
import time
import logging
import asyncio
import pandas as pd
import random
from dotenv import load_dotenv

# Importações locais
from providers import OpenAIProvider, AnthropicProvider, GeminiProvider, OpenAICompatibleProvider
from prompts import build_manual_prompt, build_ai_improved_prompt, build_meta_prompt
from tests import TEST_BATTERY

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

async def run_single_test(model, test: dict, category: str, prompt_text: str, semaphore: asyncio.Semaphore) -> dict:
    # Um pequeno delay aleatório ANTES de entrar no semáforo para evitar o "Thundering Herd" (estouro de manada)
    await asyncio.sleep(random.uniform(0.1, 2.0)) 
    
    async with semaphore:
        logger.info(f"Disparando -> Modelo: {model.model_name} | Teste: {test['id']} | Prompt: {category}")
        
        start_time = time.time()
        content, p_tokens, c_tokens, t_tokens, success, error_msg = await model.generate(prompt_text)
        response_time = time.time() - start_time
        
        response_length = len(content) if success else 0
        
        if success:
            logger.info(f"✅ CONCLUÍDO ({response_time:.2f}s) -> {model.model_name} | {test['id']} | {category}")
        else:
            logger.warning(f"❌ FALHA -> {model.model_name} | {test['id']} | {category} | Erro: {error_msg}")

        return {
            "Modelo": model.model_name,
            "Categoria_Prompt": category,
            "ID_Teste": test["id"],
            "Tempo_Resposta_Segundos": round(response_time, 2),
            "Tokens_Prompt": p_tokens,
            "Tokens_Completion": c_tokens,
            "Tokens_Total": t_tokens,
            "Tamanho_Resposta_Caracteres": response_length,
            "Sucesso": success,
            "Mensagem_Erro": error_msg,
            "Resposta_IA": content if success else "ERRO NA GERAÇÃO"
        }

async def run_benchmark_async():
    logger.info("Iniciando Benchmark de LLMs Assíncrono...")
    
    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append(OpenAIProvider("gpt-5.5"))
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append(AnthropicProvider("claude-opus-4-7"))
    if os.getenv("GEMINI_API_KEY"):
        models.append(GeminiProvider("gemini-3.1-pro-preview"))
    if os.getenv("XAI_API_KEY"):
        models.append(OpenAICompatibleProvider("grok-4.3", os.getenv("XAI_API_KEY"), "https://api.x.ai/v1"))
    if os.getenv("DEEPSEEK_API_KEY"):
        models.append(OpenAICompatibleProvider("deepseek-v4-pro", os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com"))

    if not models:
        logger.error("Nenhuma chave de API válida encontrada no .env. Abortando.")
        return

    concurrency_limit = 2 
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    tasks = []

    for model in models:
        for test in TEST_BATTERY:
            prompts = {
                "Manual": build_manual_prompt(test["task"]),
                "AI-Improved": build_ai_improved_prompt(test["task"]),
                "Meta-Prompt": build_meta_prompt(test["task"], test["example"])
            }
            for category, prompt_text in prompts.items():
                tasks.append(run_single_test(model, test, category, prompt_text, semaphore))

    logger.info(f"Disparando {len(tasks)} testes simultaneamente...")
    results = await asyncio.gather(*tasks)

    logger.info("Todos os testes finalizados! Exportando dados...")
    df = pd.DataFrame(results)
    
    output_filename = "resultados_benchmark_llms_async.csv"
    try:
        # A coluna 'Resposta_IA' ficará no final do CSV, contendo o texto completo gerado
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Sucesso! Arquivo '{output_filename}' gerado com as respostas das IAs inclusas.")
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo CSV: {e}")

if __name__ == "__main__":
    asyncio.run(run_benchmark_async())