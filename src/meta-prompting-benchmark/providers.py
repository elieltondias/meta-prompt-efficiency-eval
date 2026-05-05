import os
import logging
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt

# Importação dos SDKs das APIs (Versões Async)
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.genai as genai

logger = logging.getLogger(__name__)

class LLMProvider:
    """Classe base (Interface) para os provedores de LLM."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    @retry(wait=wait_exponential(multiplier=2, min=5, max=65), stop=stop_after_attempt(6))
    async def _call_api(self, prompt: str):
        """Método interno que realiza a chamada à API de forma assíncrona."""
        raise NotImplementedError()

    async def generate(self, prompt: str) -> tuple:
        """Retorna: (resposta, prompt_tokens, completion_tokens, total_tokens, sucesso, erro_msg)"""
        try:
            return await self._call_api(prompt)
        except Exception as e:
            logger.error(f"Falha definitiva no modelo {self.model_name}: {str(e)}")
            return "", 0, 0, 0, False, str(e)


class OpenAIProvider(LLMProvider):
    """Provedor OpenAI Assíncrono."""
    def __init__(self, model_name="gpt-5.5"):
        super().__init__(model_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no .env")
        
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        
        self.client = AsyncOpenAI(api_key=api_key, http_client=http_client)

    @retry(wait=wait_exponential(multiplier=2, min=5, max=65), stop=stop_after_attempt(6))
    async def _call_api(self, prompt: str):
        is_reasoning_model = any(x in self.model_name.lower() for x in ["o1", "o3"])
        
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }

        if is_reasoning_model:
            kwargs["max_completion_tokens"] = 4096
        else:
            kwargs["temperature"] = 0.7
            kwargs["max_tokens"] = 2048

        response = await self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        usage = response.usage
        return content, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, True, ""


class AnthropicProvider(LLMProvider):
    """Provedor para Anthropic (Claude) Assíncrono."""
    def __init__(self, model_name="claude-opus-4-7"):
        super().__init__(model_name)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY não encontrada no .env")
            
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(wait=wait_exponential(multiplier=2, min=5, max=65), stop=stop_after_attempt(6))
    async def _call_api(self, prompt: str):
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.content[0].text
        p_tokens = response.usage.input_tokens
        c_tokens = response.usage.output_tokens
        return content, p_tokens, c_tokens, p_tokens + c_tokens, True, ""


class GeminiProvider(LLMProvider):
    """Provedor para Google (Gemini) usando o SDK google-genai Assíncrono."""
    def __init__(self, model_name="gemini-3.1-pro-preview"):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada no .env")
            
        self.client = genai.Client(api_key=api_key)

    @retry(wait=wait_exponential(multiplier=2, min=5, max=65), stop=stop_after_attempt(6))
    async def _call_api(self, prompt: str):
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4096
            )
        )
        content = response.text
        
        usage = response.usage_metadata
        p_tokens = getattr(usage, 'prompt_token_count', 0) if usage else 0
        c_tokens = getattr(usage, 'candidates_token_count', 0) if usage else 0
        t_tokens = getattr(usage, 'total_token_count', 0) if usage else 0
        
        return content, p_tokens, c_tokens, t_tokens, True, ""


class OpenAICompatibleProvider(OpenAIProvider):
    """Provedor genérico para APIs compatíveis com OpenAI Assíncrono."""
    def __init__(self, model_name: str, api_key: str, base_url: str):
        LLMProvider.__init__(self, model_name)
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)