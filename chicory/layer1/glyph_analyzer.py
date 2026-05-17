"""Glyph-aware text analysis — scans content for concept words and maps to glyphs.

Mirrors GPT-GU's ``analyze_text`` workflow using the hardcoded lexicon so
glyph definitions always flow through memory content, not just lattice placement.

No ByT5 model required — works entirely from glyph_lexicon.py.
"""

from __future__ import annotations

import re
from typing import Any

from chicory.layer1.glyph_lexicon import (
    DICT2GLYPH,
    GLYPH2DICT,
    PAIR_TO_TEXT,
    concept_for_glyph,
    glyph_for_concept,
)

# ── Pair detection patterns ───────────────────────────────────────────
# Matches "X vs Y", "X and Y", "X or Y", "X versus Y", "between X and Y"

_PAIR_TEMPLATES = [
    r"(\b\w+)\s+(?:vs\.?|versus)\s+(\w+)\b",
    r"between\s+(\w+)\s+and\s+(\w+)\b",
    r"(\b\w+)\s+(?:↔|⇔|<->)\s+(\w+)\b",
]

# Build case-insensitive lookup: lowercase concept → (proper_name, glyph)
_LOWER_LOOKUP: dict[str, tuple[str, str]] = {}
for _concept, _glyph in DICT2GLYPH.items():
    _LOWER_LOOKUP[_concept.lower()] = (_concept, _glyph)

# Multi-word concepts sorted longest-first for greedy matching
_MULTI_WORD = sorted(
    [(k, v) for k, v in DICT2GLYPH.items() if len(k.split()) > 1],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Single-word concepts
_SINGLE_WORD = {k: v for k, v in DICT2GLYPH.items() if len(k.split()) == 1}


def analyze_text(text: str) -> dict[str, Any]:
    """Scan text for concept words, map to glyphs, detect pairs.

    Returns a dict with:
      - ``words``: list of {word, glyph} for each concept found
      - ``pairs``: list of {text, glyphs, concepts} for detected pairs
      - ``glyph_line``: glyph-annotated version of the text
      - ``formula``: compact glyph sequence of all found concepts
    """
    if not text:
        return {"words": [], "pairs": [], "glyph_line": None, "formula": None}

    words: list[dict[str, str]] = []
    seen_lower: set[str] = set()

    # Phase 1: Multi-word matches (longest first to avoid partial hits)
    for concept, glyph in _MULTI_WORD:
        if concept.lower() in text.lower() and concept.lower() not in seen_lower:
            words.append({"word": concept, "glyph": glyph})
            seen_lower.add(concept.lower())
            # Mark component words as seen
            for part in concept.lower().split():
                seen_lower.add(part)

    # Phase 2: Single-word matches
    individual = re.findall(r"\b[A-Za-z]+\b", text)
    for word in individual:
        low = word.lower()
        if low in seen_lower:
            continue
        hit = _LOWER_LOOKUP.get(low)
        if hit:
            concept_name, glyph = hit
            words.append({"word": concept_name, "glyph": glyph})
            seen_lower.add(low)

    # Phase 3: Detect pairs
    pairs: list[dict[str, Any]] = []
    for pat in _PAIR_TEMPLATES:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            a = m.group(1).strip(" .,:;-")
            b = m.group(2).strip(" .,:;-")
            if not a or not b:
                continue

            ga = glyph_for_concept(a)
            gb = glyph_for_concept(b)
            if ga and gb:
                pair_key = (ga, gb)
                pair_concepts = PAIR_TO_TEXT.get(pair_key)
                pairs.append({
                    "text": f"{a} vs {b}",
                    "glyphs": f"{ga}/{gb}",
                    "concepts": pair_concepts,
                    "is_known_pair": pair_concepts is not None,
                })

    # Phase 4: Build compact word:glyph mapping
    glyph_line = None
    if words:
        glyph_line = " ".join(f'{w["word"]}:{w["glyph"]}' for w in words)

    # Phase 5: Compact formula
    formula = None
    if words:
        formula = " ".join(w["glyph"] for w in words)

    return {
        "words": words,
        "pairs": pairs,
        "glyph_line": glyph_line,
        "formula": formula,
    }


def extract_glyph_tags(text: str) -> list[str]:
    """Extract concept names found in text, suitable for use as tags.

    Returns lowercase-hyphenated tag names (e.g. "absolute-change").
    """
    result = analyze_text(text)
    tags: list[str] = []
    for w in result["words"]:
        tag = w["word"].lower().replace(" ", "-")
        if tag not in tags:
            tags.append(tag)
    return tags


def glyph_summary(text: str) -> str | None:
    """Return a compact glyph summary line, or None if no concepts found.

    Format: ``concept1(glyph1) concept2(glyph2) ...``
    """
    result = analyze_text(text)
    if not result["words"]:
        return None

    parts: list[str] = []
    for w in result["words"]:
        parts.append(f'{w["word"]}({w["glyph"]})')

    line = " ".join(parts)

    # Append pair annotations
    for p in result["pairs"]:
        if p["is_known_pair"]:
            line += f'  [{p["glyphs"]}]'

    return line


def annotate_for_storage(text: str) -> dict[str, Any]:
    """Analyze text and return storage-ready metadata.

    Returns:
      - ``glyph_tags``: list of tag names derived from detected concepts
      - ``glyph_concepts``: list of {concept, glyph} found in content
      - ``glyph_pairs``: list of detected glyph pairs
      - ``glyph_line``: annotated text with glyphs inline
      - ``formula``: compact glyph sequence
    """
    result = analyze_text(text)
    return {
        "glyph_tags": extract_glyph_tags(text),
        "glyph_concepts": [
            {"concept": w["word"], "glyph": w["glyph"]}
            for w in result["words"]
        ],
        "glyph_pairs": result["pairs"],
        "glyph_line": result["glyph_line"],
        "formula": result["formula"],
    }
