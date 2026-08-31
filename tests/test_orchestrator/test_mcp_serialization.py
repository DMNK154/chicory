r"""Response serialization for the MCP layer.

The glyph legend exists so a reader can decode the glyph characters that
appear in retrieval results. It was inert for its whole life: json.dumps
defaults to ensure_ascii=True, so every glyph became a \uXXXX escape
BEFORE the injector scanned the string for glyph characters. It found
none, appended nothing, and clients received bare codepoints with the
decoder ring that exists for them never sent. Nothing raised; the feature
simply did not run.
"""

from __future__ import annotations

import inspect
import json

import pytest

from chicory.mcp import server
from chicory.mcp.server import _ensure_glyph_legend, _inject_glyph_definitions


def _a_real_glyph():
    legend = _ensure_glyph_legend()
    if not legend:
        pytest.skip("no glyph lexicon in this install")
    return next(iter(legend))


def test_injector_finds_glyphs_in_unescaped_json():
    glyph = _a_real_glyph()
    payload = json.dumps(
        {"glyph_reading": f"Reason:{glyph}"}, default=str, ensure_ascii=False,
    )
    assert "[glyph definitions:" in _inject_glyph_definitions(payload)


def test_ascii_escaped_json_defeats_the_injector():
    """Pins why ensure_ascii=False is required, so restoring the default
    cannot silently disable the legend again."""
    glyph = _a_real_glyph()
    escaped = json.dumps({"glyph_reading": f"Reason:{glyph}"}, default=str)
    assert glyph not in escaped, "precondition: the glyph should be escaped"
    assert "[glyph definitions:" not in _inject_glyph_definitions(escaped)


def test_every_serialization_site_disables_ascii_escaping():
    """The tool-call path must not escape glyphs before injecting."""
    src = inspect.getsource(server)
    feeding = [
        ln.strip() for ln in src.splitlines()
        if "json.dumps(" in ln and ("result," in ln or "stats," in ln)
    ]
    assert feeding, "no serialization sites found — this test is stale"
    for ln in feeding:
        assert "ensure_ascii=False" in ln, (
            f"site escapes glyphs before the legend scan: {ln}"
        )
