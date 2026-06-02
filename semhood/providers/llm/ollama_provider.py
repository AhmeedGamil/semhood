"""Ollama local LLM provider — fully offline, no API keys needed."""

from __future__ import annotations

import httpx

from semhood.providers.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Ollama local LLM (Llama, DeepSeek Coder, CodeLlama, Mistral, etc.).

    Runs fully offline via localhost. No API keys, no cost.
    Requires Ollama to be running: https://ollama.ai
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "deepseek-coder-v2"):
        self._url = url.rstrip("/")
        self._model = model

    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
