"""OpenRouter LLM provider — one key, every major model.

OpenRouter (https://openrouter.ai) exposes Anthropic, OpenAI, Google,
Meta, Mistral, etc. behind a single OpenAI-compatible API. We just point
the OpenAI client at their base URL and forward the recommended headers.

Pick a model with the standard ``provider/model`` slug, e.g.:
  - "anthropic/claude-sonnet-4"
  - "openai/gpt-4o"
  - "google/gemini-2.5-pro"
  - "meta-llama/llama-3.3-70b-instruct"

See https://openrouter.ai/models for the full list.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from semhood.providers.llm.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter — multi-model gateway via the OpenAI client."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        base_url: str | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
    ):
        """
        Args:
            api_key:      OpenRouter API key (``sk-or-...``).
            model:        ``provider/model`` slug.
            base_url:     Override for self-hosted / proxy setups.
            http_referer: Optional ``HTTP-Referer`` header — OpenRouter uses
                          this for attribution and rate-limit accounting.
            x_title:      Optional ``X-Title`` header for the same purpose.
        """
        self._model = model

        # Headers OpenRouter recommends but doesn't require.
        default_headers: dict[str, str] = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if x_title:
            default_headers["X-Title"] = x_title

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_headers=default_headers or None,
        )

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
