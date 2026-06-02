"""OpenAI GPT LLM provider."""

from __future__ import annotations

from openai import AsyncOpenAI

from semhood.providers.llm.base import LLMProvider


class OpenAILLMProvider(LLMProvider):
    """OpenAI GPT models (GPT-4o, GPT-4-turbo, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content or ""
