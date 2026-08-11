"""Splitting documents into retrievable chunks.

Chunking is the decision that quietly caps a RAG system's ceiling. Split too
small and an answer is spread across three chunks, none of which retrieves. Too
large and the embedding averages five topics into a vector that matches nothing
in particular, while the reader drowns the answer in context.

The strategy here is structure-first: split on Markdown headings, because a
heading is the author's own statement about where one idea ends. Only sections
too large to embed usefully are split further, on paragraph then sentence
boundaries, with an overlap so a fact sitting on a seam survives in one piece.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
PARAGRAPH = re.compile(r"\n\s*\n")
# The Cyrillic range is deliberate: sentence boundaries have to work on Russian
# documentation too, and a Latin-only class silently treats a whole Russian
# paragraph as one unsplittable sentence.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])")  # noqa: RUF001


@dataclass(slots=True)
class Chunk:
    """One retrievable passage, and enough provenance to cite it."""

    text: str
    source: str
    section: str = ""
    ordinal: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """What gets shown under an answer."""
        return f"{self.source}#{self.section}" if self.section else self.source

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "ordinal": self.ordinal,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata,
        }


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split oversized text on the least damaging boundary available.

    Paragraphs first, then sentences, then a hard cut. Each fallback loses more
    context than the last, which is exactly why they are tried in that order.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for paragraph in PARAGRAPH.split(text):
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            pieces.append(buffer)
        if len(paragraph) <= max_chars:
            buffer = paragraph
            continue

        # One paragraph is itself too long: fall back to sentences.
        buffer = ""
        for sentence in SENTENCE.split(paragraph):
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                # A single sentence longer than the limit gets cut hard. Rare
                # outside of generated text, and better than dropping it.
                while len(sentence) > max_chars:
                    pieces.append(sentence[:max_chars])
                    sentence = sentence[max_chars - overlap :]
                buffer = sentence

    if buffer:
        pieces.append(buffer)

    return _apply_overlap(pieces, overlap)


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prefix each piece with the tail of the previous one.

    A fact that straddles a boundary — a definition and the sentence that uses
    it — is otherwise split across two chunks that each retrieve poorly.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    for previous, current in itertools.pairwise(pieces):
        tail = previous[-overlap:].lstrip()
        out.append(f"{tail} {current}" if tail else current)
    return out


def chunk_markdown(
    text: str,
    source: str,
    max_chars: int = 1200,
    overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split on headings, then size."""
    metadata = metadata or {}
    matches = list(HEADING.finditer(text))

    if not matches:
        sections = [("", text, 0)]
    else:
        sections = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble, 0))
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((match.group(2).strip(), text[start:end].strip(), start))

    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body, offset in sections:
        if not body.strip():
            continue
        for piece in _split_long(body, max_chars, overlap):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    section=heading,
                    ordinal=ordinal,
                    char_start=offset,
                    char_end=offset + len(piece),
                    metadata=dict(metadata),
                )
            )
            ordinal += 1

    return chunks


def read_document(path: Path) -> tuple[str, dict]:
    """Read one file to text, with format-specific handling for PDF."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PDF support needs: pip install askdocs[pdf]") from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages), {"pages": len(pages)}

    return path.read_text(encoding="utf-8", errors="replace"), {}


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".pdf"}


def chunk_file(
    path: Path, max_chars: int = 1200, overlap: int = 120
) -> list[Chunk]:
    text, metadata = read_document(path)
    return chunk_markdown(text, str(path), max_chars, overlap, metadata)


def find_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
