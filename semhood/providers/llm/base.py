"""Abstract interface for LLM providers."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for language model operations.

    Used for:
      - Logic summary generation (Smart strategy)
      - Developer query generation (Smart strategy)
      - Query expansion (query pipeline)
      - Final answer generation (query pipeline)
    """

    @abstractmethod
    async def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        """Generate a text completion."""
        ...

    async def generate_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> dict:
        """
        Generate a JSON-structured completion.

        Default implementation calls generate() and parses the result.
        Override in subclasses for native JSON mode support.
        """
        raw = await self.generate(system_prompt, user_prompt, max_tokens)

        # Try to extract JSON from the response
        raw = raw.strip()
        if raw.startswith("```"):
            # Strip markdown code fences
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response, returning empty dict")
            return {}
