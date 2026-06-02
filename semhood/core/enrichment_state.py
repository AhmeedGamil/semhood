"""Enrichment lifecycle states.

Single source of truth for the four states a chunk can be in with respect
to LLM enrichment. Use these constants everywhere instead of bare strings
to avoid typos.
"""

from __future__ import annotations

PENDING = "pending"
FRESH = "fresh"
FAILED = "failed"
SKIPPED_TRIVIAL = "skipped_trivial"

ALL_STATES = (PENDING, FRESH, FAILED, SKIPPED_TRIVIAL)
