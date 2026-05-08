"""Extractive document summarizer for reference-tier ingestion.

Produces a whole-document summary by extracting structural elements:
headings, first paragraph, section leads, key terms, and document stats.
No LLM required — deterministic and fast.
"""

from __future__ import annotations

import re


def summarize_document(text: str, rel_path: str, max_chars: int = 2000) -> str:
    """Generate an extractive summary covering the entire document.

    Extracts:
    - Document stats (word count, section count)
    - Heading structure
    - Opening paragraph
    - First sentence of each section
    - Key list items or definitions

    Returns a self-contained summary suitable for embedding.
    """
    if not text.strip():
        return f"[{rel_path}]\n\nEmpty document."

    lines = text.split("\n")
    word_count = len(text.split())

    headings = _extract_headings(lines)
    first_para = _extract_first_paragraph(text)
    section_leads = _extract_section_leads(lines)
    key_items = _extract_key_items(lines)

    parts: list[str] = [f"[{rel_path}]"]

    stats_parts = [f"{word_count} words"]
    if headings:
        stats_parts.append(f"{len(headings)} sections")
    parts.append(f"Document summary: {', '.join(stats_parts)}")

    if headings:
        heading_text = " → ".join(h["text"] for h in headings[:12])
        parts.append(f"Structure: {heading_text}")

    if first_para:
        parts.append(first_para[:600])

    if section_leads:
        leads_text = " ".join(section_leads[:6])
        if leads_text != first_para[:len(leads_text)]:
            parts.append(f"Key points: {leads_text}")

    if key_items:
        parts.append("Items: " + "; ".join(key_items[:8]))

    summary = "\n\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars - 3] + "..."
    return summary


def _extract_headings(lines: list[str]) -> list[dict]:
    results = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            if text:
                results.append({"level": level, "text": text})
    return results


def _extract_first_paragraph(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("#"):
            continue
        if len(p) < 20:
            continue
        return p
    return ""


def _extract_section_leads(lines: list[str]) -> list[str]:
    """First meaningful sentence after each heading."""
    leads: list[str] = []
    after_heading = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            after_heading = True
            continue
        if after_heading and stripped and not stripped.startswith("#"):
            sentence = _first_sentence(stripped)
            if sentence and len(sentence) > 15:
                leads.append(sentence)
            after_heading = False
    return leads


def _extract_key_items(lines: list[str]) -> list[str]:
    """Extract bulleted/numbered list items and definition-like lines."""
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-*•]\s+\*?\*?[A-Z]", stripped):
            item = re.sub(r"^[-*•]\s+", "", stripped)
            item = item.rstrip(".")
            if 10 < len(item) < 120:
                items.append(item)
        elif re.match(r"^\d+[.)]\s+", stripped):
            item = re.sub(r"^\d+[.)]\s+", "", stripped)
            item = item.rstrip(".")
            if 10 < len(item) < 120:
                items.append(item)
    return items


def _first_sentence(text: str) -> str:
    match = re.match(r"^(.+?[.!?])\s", text)
    if match:
        return match.group(1)
    if len(text) < 150:
        return text
    return text[:150] + "..."
