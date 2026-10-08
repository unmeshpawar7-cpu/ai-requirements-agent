"""Text normalisation applied to every extracted document before chunking.

The goal is stable, comparable text (for checksums, embeddings and citations) without losing
the structure hints the chunker uses (headings, list items, paragraph breaks).
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_WS = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_BULLET = re.compile(r"^\s*[\u2022\u2023\u25aa\u25cf\u25e6\u2043\u2219\u00b7*\u2013\u2014]\s+")
_SMART_QUOTES = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2032": "'", "\u2033": '"'}
)


def normalise_text(text: str) -> str:
    """Return a canonical form of `text`.

    Steps: Unicode NFKC, unified newlines, control characters removed, smart quotes folded to
    ASCII, PDF-style hyphenated line breaks re-joined, inline whitespace collapsed, bullet glyphs
    turned into `- `, trailing spaces stripped and runs of blank lines collapsed to one.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = text.translate(_SMART_QUOTES)
    text = _HYPHENATED_BREAK.sub(r"\1\2", text)

    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _INLINE_WS.sub(" ", raw_line).strip()
        line = _BULLET.sub("- ", line)
        lines.append(line)
    text = "\n".join(lines)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()
