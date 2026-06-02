"""
Evaluator — runs a dataset of test cases through the query pipeline
and measures retrieval quality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from semhood.core.query_pipeline import QueryPipeline
from semhood.evaluation.metrics import source_hit_rate, summarize_results

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs RAGAS-style evaluation against a test dataset."""

    def __init__(self, pipeline: QueryPipeline):
        self._pipeline = pipeline

    async def evaluate(self, dataset_path: str, verbose: bool = False) -> dict:
        """
        Run evaluation on a JSON dataset.

        Dataset format:
        [
          {
            "question": "How does user authentication work?",
            "expected_sources": ["app/services/auth.py"],
            "expected_answer_contains": ["JWT", "token"]
          },
          ...
        ]

        Returns aggregate metrics.
        """
        data = json.loads(Path(dataset_path).read_text())
        results: list[dict] = []

        for i, case in enumerate(data):
            question = case["question"]
            expected_sources = case.get("expected_sources", [])
            expected_keywords = case.get("expected_answer_contains", [])

            logger.info("Evaluating [%d/%d]: %s", i + 1, len(data), question[:80])

            result_data = await self._pipeline.query(question)

            # Source hit rate
            retrieved_files = [s.get("file", "") for s in result_data.sources]
            hit_rate = source_hit_rate(retrieved_files, expected_sources)

            # Keyword presence in answer
            keyword_hits = 0
            if expected_keywords:
                answer_lower = result_data.answer.lower()
                for kw in expected_keywords:
                    if kw.lower() in answer_lower:
                        keyword_hits += 1
                keyword_rate = keyword_hits / len(expected_keywords)
            else:
                keyword_rate = 1.0

            case_result = {
                "question": question,
                "source_hit_rate": hit_rate,
                "keyword_rate": keyword_rate,
                "sources_found": len(result_data.sources),
            }
            results.append(case_result)

            if verbose:
                logger.info(
                    "  → hit_rate=%.2f, keyword_rate=%.2f, sources=%d",
                    hit_rate, keyword_rate, len(result_data.sources),
                )

        summary = summarize_results(results)
        summary["total_cases"] = len(results)

        return summary
