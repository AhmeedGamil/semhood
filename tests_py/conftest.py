"""Shared pytest fixtures."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make the package importable when running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def event_loop():
    """Per-test event loop so async tests don't leak state."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
