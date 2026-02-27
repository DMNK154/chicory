"""Smart document chunking — split by sections/paragraphs with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of document text with metadata."""

    text: str
    index: int  # Position in the original document
    total: int  # Total number of chunks
    source_file: str
    section_title: str | None = None


# Default: ~500 tokens per chunk, ~100 token overlap
DEFAULT_CHUNK_SIZE = 2000  # characters (~500 tokens)
DEFAULT_OVERLAP = 400  # characters (~100 tokens)
MIN_CHUNK_SIZE = 200  # Don't create tiny chunks

# Embedding model token limit: all-MiniLM-L6-v2 supports 256 tokens (~1000 chars).
# Chunks for embedding must stay within this limit.
MAX_EMBED_CHARS = 1000


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a document into chunks using smart boundaries.

    Strategy:
    1. Try to split on markdown/section headers first
    2. Within sections, split on paragraph boundaries
    3. If paragraphs are still too long, split on sentence boundaries
    4. Last resort: split on word boundaries at chunk_size
    """
    if not text or not text.strip():
        return []

    # If the whole document fits in one chunk, return it as-is
    if len(text) <= chunk_size:
        return [Chunk(
            text=text.strip(),
            index=0,
            total=1,
            source_file=source_file,
        )]

    # Try section-based splitting first
    sections = _split_into_sections(text)

    chunks: list[Chunk] = []

    for section_title, section_text in sections:
        if len(section_text) <= chunk_size:
            chunks.append(Chunk(
                text=section_text.strip(),
                index=len(chunks),
                total=0,  # Will be updated
                source_file=source_file,
                section_title=section_title,
            ))
        else:
            # Split section into paragraph-based chunks with overlap
            sub_chunks = _split_with_overlap(
                section_text, chunk_size, overlap, source_file, section_title
            )
            for sc in sub_chunks:
                sc.index = len(chunks)
                chunks.append(sc)

    # Update totals
    for c in chunks:
        c.total = len(chunks)

    return chunks


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text on markdown headers or obvious section boundaries."""
    # Match markdown headers (# Header) or uppercase-only lines followed by content
    header_pattern = re.compile(
        r"^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}[A-Z])$", re.MULTILINE
    )

    matches = list(header_pattern.finditer(text))

    if not matches:
        return [(None, text)]

    sections = []

    # Content before first header
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append((None, pre))

    for i, match in enumerate(matches):
        title = match.group().strip().lstrip("#").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))

    return sections if sections else [(None, text)]


def _split_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
    source_file: str,
    section_title: str | None,
) -> list[Chunk]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Chunk] = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(Chunk(
                text=current.strip(),
                index=0,
                total=0,
                source_file=source_file,
                section_title=section_title,
            ))

            # Keep overlap from the end of current chunk
            if overlap > 0 and len(current) > overlap:
                # Find a sentence boundary near the overlap point
                overlap_text = current[-(overlap):]
                sentence_break = overlap_text.find(". ")
                if sentence_break > 0:
                    overlap_text = overlap_text[sentence_break + 2 :]
                current = overlap_text + "\n\n" + para
            else:
                current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

        # Handle single paragraphs that are too long
        if len(current) > chunk_size * 1.5:
            sub = _split_on_sentences(current, chunk_size, overlap)
            for s in sub[:-1]:
                chunks.append(Chunk(
                    text=s.strip(),
                    index=0,
                    total=0,
                    source_file=source_file,
                    section_title=section_title,
                ))
            current = sub[-1] if sub else ""

    if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(Chunk(
            text=current.strip(),
            index=0,
            total=0,
            source_file=source_file,
            section_title=section_title,
        ))
    elif current.strip() and chunks:
        # Append small remainder to last chunk
        chunks[-1].text += "\n\n" + current.strip()

    return chunks


def _split_on_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last resort: split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            parts.append(current)
            # Overlap
            if overlap > 0:
                current = current[-(overlap):] + " " + sentence
            else:
                current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        parts.append(current)

    return parts


def chunk_text_for_embedding(
    text: str, max_chars: int = MAX_EMBED_CHARS
) -> list[str]:
    """Split text into chunks that fit within embedding model token limits.

    Returns a list of text chunks. Short text returns a single-element list.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current.strip():
        chunks.append(current.strip())

    # Handle sentences that individually exceed max_chars: split on word boundaries
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            words = chunk.split()
            part = ""
            for word in words:
                if part and len(part) + len(word) + 1 > max_chars:
                    final.append(part.strip())
                    part = word
                else:
                    part = (part + " " + word).strip() if part else word
            if part.strip():
                final.append(part.strip())

    return final if final else [text[:max_chars]]
