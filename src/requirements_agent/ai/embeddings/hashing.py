"""Deterministic, offline feature-hashing embeddings.

This is a *lexical* embedding: it captures shared vocabulary (words, word pairs and character
trigrams, which give some robustness to inflection such as "approve"/"approval"), not meaning.
"invoice" and "bill" are unrelated to it. It exists so the whole RAG pipeline runs and is
testable without network access or API keys; use `EMBEDDING_PROVIDER=openai` for semantic
retrieval.

Features and weights:
* word unigrams (stop words removed) ............ 1.0
* word bigrams .................................. 0.5
* character trigrams of each word (with `^`/`$`) .. 0.25

Each feature is hashed with BLAKE2b into one of `dimension` buckets with a hash-derived sign
(the "hashing trick" with signed buckets, which keeps collisions unbiased), term frequencies
are dampened with `1 + log(tf)`, and the vector is L2-normalised so cosine similarity is a dot
product.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "all",
        "any",
        "each",
        "also",
        "not",
        "no",
        "only",
        "own",
        "same",
        "very",
        "via",
        "per",
    ]
)
_WEIGHTS = {"uni": 1.0, "bi": 0.5, "tri": 0.25}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP_WORDS and len(t) > 1]


class HashingEmbeddingProvider:
    name = "hashing"

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension
        self.model = "hashing-v1"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for feature, weight in self._features(text).items():
            index, sign = self._bucket(feature)
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()

    def _features(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        counts: Counter[str] = Counter()
        for token in tokens:
            counts[f"u:{token}"] += 1
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                counts[f"c:{padded[i : i + 3]}"] += 1
        for left, right in pairwise(tokens):
            counts[f"b:{left} {right}"] += 1

        features: dict[str, float] = {}
        for feature, tf in counts.items():
            kind = {"u": "uni", "b": "bi", "c": "tri"}[feature[0]]
            features[feature] = _WEIGHTS[kind] * (1.0 + math.log(tf))
        return features

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimension, (1.0 if (value >> 63) & 1 else -1.0)
