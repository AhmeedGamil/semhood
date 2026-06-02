"""Anthropic Claude LLM provider."""

from __future__ import annotations

import anthropic

from semhood.providers.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Claude (Anthropic) — used for enrichment, query expansion, and answer generation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return "".join(
            block.text for block in response.content if block.type == "text"
        )
