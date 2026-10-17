"""Retrieval quality metrics.

Relevance is judged per *chunk*, keyed by ``(filename, page_number)``. A
retrieved chunk is relevant iff its key appears in the expected set. These are
deterministic, transparent metrics — not LLM judgments.
"""

RelevanceKey = tuple[str, int]


def _hits(retrieved_keys: list[RelevanceKey], relevant: set[RelevanceKey], k: int) -> int:
    return sum(1 for key in retrieved_keys[:k] if key in relevant)


def recall_at_k(retrieved_keys: list[RelevanceKey], relevant: set[RelevanceKey], k: int) -> float:
    """Fraction of expected relevant keys found in the top-k results."""
    if not relevant:
        return 0.0
    found = {key for key in retrieved_keys[:k] if key in relevant}
    return len(found) / len(relevant)


def precision_at_k(retrieved_keys: list[RelevanceKey], relevant: set[RelevanceKey], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        return 0.0
    return _hits(retrieved_keys, relevant, k) / k


def reciprocal_rank(retrieved_keys: list[RelevanceKey], relevant: set[RelevanceKey]) -> float:
    """1 / rank of the first relevant result (0 if none)."""
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0
