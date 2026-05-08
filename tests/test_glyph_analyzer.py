"""Tests for glyph content analysis — mirrors GPT-GU's analyze_text."""

from __future__ import annotations

from chicory.layer1.glyph_analyzer import (
    analyze_text,
    annotate_for_storage,
    extract_glyph_tags,
    glyph_summary,
)


class TestAnalyzeText:
    def test_single_concept(self):
        result = analyze_text("Memory is the foundation of identity")
        words = {w["word"] for w in result["words"]}
        assert "Memory" in words
        assert "Foundation" in words

    def test_multi_word_concept(self):
        result = analyze_text("The Absolute Change was dramatic")
        words = {w["word"] for w in result["words"]}
        assert "Absolute Change" in words
        # "change" should NOT appear as separate since it's part of multi-word
        assert "Change" not in words

    def test_case_insensitive(self):
        result = analyze_text("hope and joy are related")
        words = {w["word"] for w in result["words"]}
        assert "Hope" in words
        assert "Joy" in words

    def test_pair_detection(self):
        result = analyze_text("Growth vs Stagnation is a key tension")
        assert len(result["pairs"]) > 0
        pair = result["pairs"][0]
        assert pair["is_known_pair"]
        assert "⟆" in pair["glyphs"]
        assert "⍭" in pair["glyphs"]

    def test_glyph_line(self):
        result = analyze_text("Memory guides the way home")
        assert result["glyph_line"] is not None
        assert "☍" in result["glyph_line"]
        assert "⌂" in result["glyph_line"]

    def test_formula(self):
        result = analyze_text("Hope and Joy")
        assert result["formula"] is not None
        assert "⎃" in result["formula"]
        assert "⧫" in result["formula"]

    def test_empty_text(self):
        result = analyze_text("")
        assert result["words"] == []
        assert result["pairs"] == []
        assert result["glyph_line"] is None

    def test_no_concepts(self):
        result = analyze_text("The quick brown fox jumped over the lazy dog")
        assert result["words"] == []


class TestExtractGlyphTags:
    def test_extracts_tags(self):
        tags = extract_glyph_tags("Memory and Hope together create Continuity")
        assert "memory" in tags
        assert "hope" in tags
        assert "continuity" in tags

    def test_multi_word_tag(self):
        tags = extract_glyph_tags("Absolute Change is inevitable")
        assert "absolute-change" in tags

    def test_no_duplicates(self):
        tags = extract_glyph_tags("Memory Memory Memory")
        assert tags.count("memory") == 1


class TestGlyphSummary:
    def test_summary(self):
        s = glyph_summary("Joy and Hope")
        assert s is not None
        assert "Joy" in s
        assert "⧫" in s
        assert "Hope" in s
        assert "⎃" in s

    def test_no_concepts(self):
        assert glyph_summary("nothing here") is None


class TestAnnotateForStorage:
    def test_full_annotation(self):
        meta = annotate_for_storage(
            "Consciousness is the hard problem of Psyche and Reason"
        )
        assert len(meta["glyph_tags"]) > 0
        assert len(meta["glyph_concepts"]) > 0
        assert any(c["concept"] == "Psyche" for c in meta["glyph_concepts"])
        assert any(c["concept"] == "Reason" for c in meta["glyph_concepts"])
