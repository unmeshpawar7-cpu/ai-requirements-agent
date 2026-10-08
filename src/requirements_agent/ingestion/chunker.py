"""Heading- and paragraph-aware, token-budgeted chunking with overlap.

Algorithm
---------
1. The normalised text is parsed into *units* with exact character offsets:
   heading lines (Markdown `#`, numbered `3.2 Title` or ALL-CAPS standalone lines) and body
   sentences (paragraphs and list items are split into sentences; a sentence longer than the
   budget is split into word windows).
2. Units are packed greedily into chunks, where a chunk is always a *contiguous span* of the
   source text (so `text == source[char_start:char_end]` and citations can be highlighted):
   * a heading always starts a new chunk (chunks never straddle sections);
   * when the next sentence would push the chunk over `max_tokens`, the chunk is closed and the
     next one starts with the trailing sentences of the previous chunk (up to
     `overlap_tokens`), so a requirement split across a boundary is still retrievable;
   * leading headings are dropped from a chunk if keeping them would break the budget (the
     section path is still recorded on the chunk).
3. Every chunk records its heading path (e.g. `"FieldOps SRS > 3 Functional > 3.2 Dispatch"`),
   used as retrieval context and shown with citations.

Invariant (tested): every chunk has `token_count <= max_tokens`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from requirements_agent.ingestion.tokens import estimate_tokens, max_chars_for_tokens

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+([A-Za-z][^\n]*)$")
_LIST_ITEM = re.compile(r"^(?:[-+]\s+|\d+[.)]\s+|[a-z][.)]\s+)")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s+[A-Z0-9\"'(\[])")
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "approx", "incl", "no", "fig", "min",
    "max", "est", "dept", "ref", "sec", "v",
}
_MAX_HEADING_CHARS = 200


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    text: str
    section_path: str
    token_count: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    kind: Literal["heading", "body"]
    section_path: str


class Chunker:
    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 50) -> None:
        if max_tokens < 20:
            raise ValueError("max_tokens must be >= 20")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Leave headroom so a heading line can usually stay attached to an oversized sentence.
        self._window_chars = max_chars_for_tokens(int(max_tokens * 0.8))

    # ------------------------------------------------------------------ public API
    def split(self, text: str) -> list[Chunk]:
        units = self._parse_units(text)
        if not units:
            return []
        spans = self._pack(text, units)
        chunks: list[Chunk] = []
        for ordinal, group in enumerate(spans):
            start, end = group[0].start, group[-1].end
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=chunk_text,
                    section_path=group[-1].section_path,
                    token_count=estimate_tokens(chunk_text),
                    char_start=start,
                    char_end=end,
                )
            )
        return chunks

    # ------------------------------------------------------------------ parsing
    def _parse_units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        heading_stack: list[tuple[int, str]] = []

        for block_start, block_end in _blocks(text):
            block = text[block_start:block_end]
            heading = _heading(block)
            if heading is not None:
                level, title = heading
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                path = " > ".join(t for _, t in heading_stack)
                units.append(_Unit(block_start, block_end, "heading", path))
                continue

            path = " > ".join(t for _, t in heading_stack)
            for seg_start, seg_end in _segments(text, block_start, block_end):
                for s_start, s_end in _sentences(text, seg_start, seg_end):
                    for w_start, w_end in self._windows(text, s_start, s_end):
                        units.append(_Unit(w_start, w_end, "body", path))
        return units

    def _windows(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        if end - start <= self._window_chars:
            return [(start, end)]
        windows: list[tuple[int, int]] = []
        win_start: int | None = None
        win_end = start
        for match in re.finditer(r"\S+", text[start:end]):
            w_start, w_end = start + match.start(), start + match.end()
            if win_start is None:
                win_start = w_start
            elif w_end - win_start > self._window_chars:
                windows.append((win_start, win_end))
                win_start = w_start
            win_end = w_end
            # A single "word" longer than the window (e.g. a base64 blob) is hard-split.
            while win_end - win_start > self._window_chars:
                windows.append((win_start, win_start + self._window_chars))
                win_start += self._window_chars
        if win_start is not None and win_end > win_start:
            windows.append((win_start, win_end))
        return windows

    # ------------------------------------------------------------------ packing
    def _tokens(self, text: str, units: list[_Unit]) -> int:
        return estimate_tokens(text[units[0].start : units[-1].end]) if units else 0

    def _pack(self, text: str, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []

        for unit in units:
            if unit.kind == "heading":
                if any(u.kind == "body" for u in current):
                    groups.append(current)
                    current = []
                current.append(unit)
                while len(current) > 1 and self._tokens(text, current) > self.max_tokens:
                    current.pop(0)
                continue

            candidate = [*current, unit]
            if self._tokens(text, candidate) <= self.max_tokens:
                current = candidate
                continue

            if any(u.kind == "body" for u in current):
                groups.append(current)
                current = self._overlap(text, current)
            # Drop carried-over / heading units from the front until the new unit fits.
            current = [*current, unit]
            while len(current) > 1 and self._tokens(text, current) > self.max_tokens:
                current.pop(0)

        if current:
            groups.append(current)
        return groups

    def _overlap(self, text: str, closed: list[_Unit]) -> list[_Unit]:
        if self.overlap_tokens == 0:
            return []
        body = [u for u in closed if u.kind == "body"]
        carry: list[_Unit] = []
        for unit in reversed(body[1:]):  # never carry the whole chunk over
            trial = [unit, *carry]
            if self._tokens(text, trial) > self.overlap_tokens:
                break
            carry = trial
        return carry


# ---------------------------------------------------------------------- helpers
def _blocks(text: str) -> list[tuple[int, int]]:
    """Blank-line separated blocks; heading lines are always split into their own block."""
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start, content = pos, line.rstrip("\n")
        pos += len(line)
        line_end = line_start + len(content)
        if not content.strip():
            if block_start is not None:
                blocks.append((block_start, block_end))
                block_start = None
            continue
        if _MD_HEADING.match(content):
            if block_start is not None:
                blocks.append((block_start, block_end))
            blocks.append((line_start, line_end))
            block_start = None
            continue
        if block_start is None:
            block_start = line_start
        block_end = line_end
    if block_start is not None:
        blocks.append((block_start, block_end))
    return blocks


def _heading(block: str) -> tuple[int, str] | None:
    if "\n" in block or len(block) > _MAX_HEADING_CHARS or block.startswith(("- ", "+ ")):
        return None
    md = _MD_HEADING.match(block)
    if md:
        return len(md.group(1)), md.group(2).strip()
    numbered = _NUMBERED_HEADING.match(block)
    if numbered and not block.rstrip().endswith((".", ":", ";", ",")) and len(block.split()) <= 12:
        return numbered.group(1).count(".") + 1, block.strip()
    letters = [c for c in block if c.isalpha()]
    if (
        len(letters) >= 3
        and all(c.isupper() for c in letters)
        and len(block) <= 80
        and not block.rstrip().endswith((".", ";", ","))
    ):
        return 1, block.strip()
    return None


def _segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a block into list items (each item with its continuation lines) or one paragraph."""
    segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    seg_end = start
    pos = start
    for line in text[start:end].split("\n"):
        line_start, line_end = pos, pos + len(line)
        pos = line_end + 1
        if _LIST_ITEM.match(line) and seg_start is not None:
            segments.append((seg_start, seg_end))
            seg_start = None
        if seg_start is None:
            seg_start = line_start
        seg_end = line_end
    if seg_start is not None:
        segments.append((seg_start, seg_end))
    return segments


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    segment = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(segment):
        boundary = match.end()
        preceding = segment[cursor : match.start()].split()
        last_word = preceding[-1].lower().rstrip(".") if preceding else ""
        if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue
        piece = segment[cursor:boundary]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            spans.append((start + cursor + lead, start + boundary))
        cursor = boundary
    tail = segment[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        spans.append((start + cursor + lead, start + cursor + len(tail.rstrip())))
    return spans
