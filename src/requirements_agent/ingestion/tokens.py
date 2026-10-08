"""Offline token estimation.

We deliberately avoid a tokenizer dependency (tiktoken downloads BPE files at runtime and is
model-specific). For English prose, ~4 characters per token is the widely quoted average for
GPT/Claude-family tokenizers; it slightly over-estimates for plain prose and under-estimates
for code/JSON. Budgets built on it therefore keep a safety margin (see docs/ai-design.md).
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimated token count: ceil(len(text) / 4); 0 for empty or whitespace-only text."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / CHARS_PER_TOKEN))


def max_chars_for_tokens(tokens: int) -> int:
    return int(tokens * CHARS_PER_TOKEN)
