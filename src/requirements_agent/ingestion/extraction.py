"""Text extraction per file type (.txt, .md, .pdf, .docx).

Extractors return plain text that preserves *structure hints* the chunker relies on:
Markdown-style `#` headings (DOCX heading styles are mapped to them), `- ` list items and
blank lines between paragraphs.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath

logger = logging.getLogger(__name__)


class DocumentFormat(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"


CONTENT_TYPES: dict[DocumentFormat, str] = {
    DocumentFormat.TEXT: "text/plain",
    DocumentFormat.MARKDOWN: "text/markdown",
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_EXTENSIONS: dict[str, DocumentFormat] = {
    ".txt": DocumentFormat.TEXT,
    ".text": DocumentFormat.TEXT,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
}

_CONTENT_TYPE_FORMATS: dict[str, DocumentFormat] = {
    "text/plain": DocumentFormat.TEXT,
    "text/markdown": DocumentFormat.MARKDOWN,
    "text/x-markdown": DocumentFormat.MARKDOWN,
    "application/pdf": DocumentFormat.PDF,
    CONTENT_TYPES[DocumentFormat.DOCX]: DocumentFormat.DOCX,
}


class ExtractionError(Exception):
    """The document could not be turned into text (corrupt, encrypted, scanned, ...)."""


class UnsupportedFormatError(ExtractionError):
    """The file type is not one of the supported formats."""


@dataclass(slots=True)
class ExtractedText:
    text: str
    format: DocumentFormat
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def detect_format(filename: str | None, content_type: str | None) -> DocumentFormat:
    """Resolve the format from the file extension, falling back to the declared MIME type."""
    if filename:
        suffix = PurePath(filename).suffix.lower()
        if suffix in _EXTENSIONS:
            return _EXTENSIONS[suffix]
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in _CONTENT_TYPE_FORMATS:
            return _CONTENT_TYPE_FORMATS[base]
    raise UnsupportedFormatError(
        "Unsupported document type; expected one of .txt, .md, .pdf, .docx"
    )


def extract_text(data: bytes, fmt: DocumentFormat) -> ExtractedText:
    if not data:
        raise ExtractionError("Document is empty")
    if fmt in (DocumentFormat.TEXT, DocumentFormat.MARKDOWN):
        return ExtractedText(text=_decode_text(data), format=fmt)
    if fmt is DocumentFormat.PDF:
        return _extract_pdf(data)
    if fmt is DocumentFormat.DOCX:
        return _extract_docx(data)
    raise UnsupportedFormatError(f"Unsupported format: {fmt}")  # pragma: no cover


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:4096]:
        raise ExtractionError("File looks binary, not text")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Legacy Windows exports are the most common non-UTF-8 input; cp1252 is a superset of
        # latin-1 for printable characters. Undefined bytes are replaced, never fatal.
        return data.decode("cp1252", errors="replace")


def _extract_pdf(data: bytes) -> ExtractedText:
    if not data.lstrip().startswith(b"%PDF-"):
        raise ExtractionError("File does not look like a PDF")
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("PDF is password protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc

    warnings: list[str] = []
    empty_pages = sum(1 for p in pages if not p.strip())
    if empty_pages:
        warnings.append(f"{empty_pages} page(s) had no extractable text")
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        raise ExtractionError("PDF contains no extractable text (scanned images need OCR)")
    return ExtractedText(
        text=text, format=DocumentFormat.PDF, page_count=len(pages), warnings=warnings
    )


def _extract_docx(data: bytes) -> ExtractedText:
    if not data.startswith(b"PK"):
        raise ExtractionError("File does not look like a DOCX (zip) document")
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several unrelated exception types
        raise ExtractionError(f"Could not read DOCX: {exc}") from exc

    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            rendered = _render_docx_paragraph(item)
            if rendered:
                blocks.append(rendered)
        elif isinstance(item, Table):
            rows = []
            for row in item.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    text = "\n\n".join(blocks)
    if not text.strip():
        raise ExtractionError("DOCX contains no text")
    return ExtractedText(text=text, format=DocumentFormat.DOCX)


def _render_docx_paragraph(paragraph: object) -> str:
    text = paragraph.text.strip()  # type: ignore[attr-defined]
    if not text:
        return ""
    style = (getattr(paragraph.style, "name", "") or "").lower()  # type: ignore[attr-defined]
    if style == "title":
        return f"# {text}"
    if style.startswith("heading"):
        # "Title" maps to "#", so "Heading N" sits one level below it.
        level_part = style.removeprefix("heading").strip()
        level = (int(level_part) if level_part.isdigit() else 1) + 1
        return f"{'#' * min(level, 6)} {text}"
    if style.startswith("list"):
        return f"- {text}"
    return text
