"""Maximal Marginal Relevance re-ranking.

MMR picks, at each step, the candidate maximising

    lambda * relevance(d) - (1 - lambda) * max_{s in selected} sim(d, s)

so near-duplicate chunks (common with overlapping chunks) do not crowd out other relevant
evidence. lambda = 1 reduces to plain relevance ranking; lambda = 0 maximises diversity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mmr_select(
    query: Sequence[float],
    candidates: Sequence[Sequence[float]],
    *,
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Return indices of up to `k` candidates in MMR order, relevance = cosine to `query`."""
    if not candidates:
        return []
    matrix = _normalise_rows(np.asarray(candidates, dtype=np.float64))
    q = np.asarray(query, dtype=np.float64)
    q_norm = float(np.linalg.norm(q))
    q = q / q_norm if q_norm else q
    return mmr_rerank((matrix @ q).tolist(), candidates, k=k, lambda_mult=lambda_mult)


def mmr_rerank(
    relevance: Sequence[float],
    candidates: Sequence[Sequence[float]],
    *,
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """MMR with precomputed relevance scores (used for multi-query retrieval, where relevance
    is the best similarity across several queries)."""
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be within [0, 1]")
    if len(relevance) != len(candidates):
        raise ValueError("relevance and candidates must have the same length")
    if k <= 0 or not candidates:
        return []

    matrix = _normalise_rows(np.asarray(candidates, dtype=np.float64))
    rel = np.asarray(relevance, dtype=np.float64)
    pairwise = matrix @ matrix.T

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: (rel[i], -i))
        else:
            redundancy = pairwise[np.ix_(remaining, selected)].max(axis=1)
            scores = lambda_mult * rel[remaining] - (1 - lambda_mult) * redundancy
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
