"""
RAGAS-style evaluation metrics for semhood.

Computes:
  - Source Hit Rate: How many expected source files were in the retrieved results
  - Other metrics can be added (faithfulness, relevancy, precision)
"""

from __future__ import annotations


def source_hit_rate(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """
    Fraction of expected sources that appear in retrieved results.

    Args:
        retrieved_sources: List of file paths from search results.
        expected_sources: List of expected file paths.

    Returns:
        Hit rate between 0.0 and 1.0.
    """
    if not expected_sources:
        return 1.0

    retrieved_set = set(retrieved_sources)
    hits = 0
    for expected in expected_sources:
        # Match by full path or by basename
        if expected in retrieved_set:
            hits += 1
        elif any(r.endswith(expected) or expected.endswith(r) for r in retrieved_set):
            hits += 1

    return hits / len(expected_sources)


def summarize_results(results: list[dict]) -> dict:
    """
    Compute aggregate statistics from a list of evaluation results.

    Each result dict should have numeric metric values.
    Returns {metric: average_value}.
    """
    if not results:
        return {}

    metrics: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(float(value))

    return {key: sum(vals) / len(vals) for key, vals in metrics.items()}
