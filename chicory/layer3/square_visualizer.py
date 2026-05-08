"""Standalone HTML/SVG rendering for square motifs."""

from __future__ import annotations

import math
from html import escape

from chicory.layer3.chain_anisotropy import ChainAnisotropyReport, ChainProbe
from chicory.layer3.cross_project_alignment import (
    AlignmentCell,
    CrossProjectAlignmentReport,
)
from chicory.layer3.cross_hidden_bridges import HiddenBridgeReport
from chicory.layer3.square_finder import DiagonalSignal, SquareMotif
from chicory.layer3.vertical_square_finder import (
    TensorVerticalSquareMotif,
    VerticalSquareMotif,
)
from chicory.layer3.zigzag_analyzer import ZigZagReport, ZigZagSample


LAYER_COLORS = {
    "cooccurrence": "#2563eb",
    "synchronicity": "#d97706",
    "semantic": "#16a34a",
    "semiotic": "#db2777",
    "glyph": "#7c3aed",
    "inhibition": "#dc2626",
}

DIAGONAL_COLORS = {
    "drawn": "#475569",
    "void": "#dc2626",
    "implied": "#f59e0b",
    "missing": "#94a3b8",
}


def render_square_motifs_html(
    motifs: list[SquareMotif],
    *,
    title: str = "Chicory Square Motifs",
) -> str:
    """Render motifs as a self-contained HTML document."""
    cards = "\n".join(_render_motif_card(motif, idx) for idx, motif in enumerate(motifs, 1))
    if not cards:
        cards = '<p class="empty">No square motifs found for this query.</p>'

    legend = "\n".join(
        f'<span class="legend-item"><span class="swatch" '
        f'style="background:{color}"></span>{escape(layer)}</span>'
        for layer, color in LAYER_COLORS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 12px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
    }}
    .node circle {{
      fill: #ffffff;
      stroke: #0f172a;
      stroke-width: 1.5;
    }}
    .node text {{
      font-size: 12px;
      fill: #0f172a;
      text-anchor: middle;
      dominant-baseline: central;
      pointer-events: none;
    }}
    .side {{
      fill: none;
      stroke-linecap: round;
    }}
    .diagonal {{
      fill: none;
      stroke-width: 2.3;
      stroke-linecap: round;
    }}
    .diag-void, .diag-implied {{
      stroke-dasharray: 8 7;
    }}
    .diag-missing {{
      stroke-dasharray: 2 7;
    }}
    .center-dot {{
      fill: #ffffff;
      stroke: #0f172a;
      stroke-width: 1.5;
    }}
    .center-label {{
      font-size: 10px;
      fill: #334155;
      text-anchor: middle;
      dominant-baseline: central;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="legend">{legend}</div>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def render_vertical_square_motifs_html(
    motifs: list[VerticalSquareMotif],
    *,
    title: str = "Chicory Vertical Square Motifs",
) -> str:
    """Render source-by-tag motifs as a self-contained HTML document."""
    cards = "\n".join(
        _render_vertical_motif_card(motif, idx)
        for idx, motif in enumerate(motifs, 1)
    )
    if not cards:
        cards = '<p class="empty">No vertical square motifs found for this query.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 12px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
    }}
    .label {{
      font-size: 12px;
      fill: #0f172a;
      dominant-baseline: central;
    }}
    .tag-label {{
      text-anchor: middle;
      font-weight: 700;
    }}
    .source-label {{
      text-anchor: end;
    }}
    .cell {{
      fill: #ffffff;
      stroke: #0f172a;
      stroke-width: 1.5;
    }}
    .grid-line {{
      stroke: #cbd5e1;
      stroke-width: 1.5;
    }}
    .relation {{
      fill: none;
      stroke-linecap: round;
    }}
    .relation-missing {{
      stroke-dasharray: 7 7;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header><h1>{escape(title)}</h1></header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def render_tensor_vertical_square_motifs_html(
    motifs: list[TensorVerticalSquareMotif],
    *,
    title: str = "Chicory Tensor-Vertical Square Motifs",
) -> str:
    """Render file-tag-by-column tensor rectangles."""
    cards = "\n".join(
        _render_tensor_vertical_motif_card(motif, idx)
        for idx, motif in enumerate(motifs, 1)
    )
    if not cards:
        cards = '<p class="empty">No tensor-vertical square motifs found for this query.</p>'

    legend = "\n".join(
        f'<span class="legend-item"><span class="swatch" '
        f'style="background:{color}"></span>{escape(layer)}</span>'
        for layer, color in LAYER_COLORS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 12px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
    }}
    .label {{
      font-size: 12px;
      fill: #0f172a;
      dominant-baseline: central;
    }}
    .tag-label {{
      text-anchor: middle;
      font-weight: 700;
    }}
    .source-label {{
      text-anchor: end;
    }}
    .cell-line {{
      stroke-linecap: round;
    }}
    .column-relation {{
      fill: none;
      stroke-linecap: round;
    }}
    .column-relation-missing {{
      stroke-dasharray: 7 7;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <div class="legend">{legend}</div>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def render_zigzag_report_html(
    report: ZigZagReport,
    *,
    title: str = "Chicory Zigzag Orientation Report",
) -> str:
    """Render a rotation-invariant zigzag metric report."""
    class_rows = "\n".join(
        _metric_bar_row(label, count, report.motif_count)
        for label, count in report.class_counts
    )
    if not class_rows:
        class_rows = '<tr><td colspan="3">No square motifs found.</td></tr>'

    signature_rows = "\n".join(
        _metric_bar_row(label, count, report.motif_count)
        for label, count in report.signature_counts
    )
    if not signature_rows:
        signature_rows = '<tr><td colspan="3">No signatures found.</td></tr>'

    pair_total = sum(count for _label, count in report.zigzag_pair_counts)
    pair_rows = "\n".join(
        _metric_bar_row(label, count, pair_total)
        for label, count in report.zigzag_pair_counts
    )
    if not pair_rows:
        pair_rows = '<tr><td colspan="3">No zigzag pairs found.</td></tr>'

    sample_cards = "\n".join(
        _render_zigzag_sample_card(sample, index)
        for index, sample in enumerate(report.samples, 1)
    )
    if not sample_cards:
        sample_cards = '<p class="empty">No examples available.</p>'

    dominant_pair = report.dominant_zigzag_pair
    pair_text = (
        f"{dominant_pair[0]} ({dominant_pair[1]})"
        if dominant_pair
        else "none"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }}
    main {{
      display: grid;
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .stat, section, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .stat {{
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.35rem;
      letter-spacing: 0;
    }}
    section {{
      padding: 14px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 7px 8px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      color: #334155;
      font-weight: 700;
      border-top: 0;
    }}
    .bar {{
      height: 12px;
      min-width: 90px;
      border-radius: 999px;
      background: var(--accent-soft);
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }}
    article {{
      padding: 12px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 0.95rem;
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="subtitle">Side colors are canonicalized across rotations and reflections before counting. The shuffled baseline preserves the global side-color frequency while scrambling square positions.</p>
  </header>
  <main>
    <div class="summary">
      {_stat("motifs", str(report.motif_count))}
      {_stat("zigzag rate", f"{report.zigzag_rate:.1%}")}
      {_stat("baseline", f"{report.baseline_zigzag_mean:.1f} +/- {report.baseline_zigzag_std:.1f}")}
      {_stat("z-score", f"{report.zigzag_z_score:.2f}")}
      {_stat("p-value", f"{report.zigzag_p_value:.3f}")}
      {_stat("vector score", f"{report.vector_zigzag_mean:.3f}")}
      {_stat("positive vectors", f"{report.vector_zigzag_positive_rate:.1%}")}
      {_stat("dominant pair", escape(pair_text))}
    </div>
    <section>
      <h2>Orientation Classes</h2>
      <table>
        <tr><th>class</th><th>count</th><th>share</th></tr>
        {class_rows}
      </table>
    </section>
    <section>
      <h2>Top Rotation-Collapsed Signatures</h2>
      <table>
        <tr><th>signature</th><th>count</th><th>share</th></tr>
        {signature_rows}
      </table>
    </section>
    <section>
      <h2>Zigzag Color Pairs</h2>
      <table>
        <tr><th>pair</th><th>count</th><th>share</th></tr>
        {pair_rows}
      </table>
    </section>
    <section>
      <h2>Example Squares</h2>
      <div class="samples">{sample_cards}</div>
    </section>
  </main>
</body>
</html>
"""


def render_chain_anisotropy_report_html(
    report: ChainAnisotropyReport,
    *,
    title: str = "Chicory Chain Anisotropy Report",
) -> str:
    """Render same-layer axis persistence against perpendicular side branches."""
    strongest = report.strongest_layer
    strongest_text = (
        f"{strongest.layer} z={strongest.z_score:.2f}"
        if strongest
        else "none"
    )
    layer_rows = "\n".join(
        _chain_layer_row(layer_report)
        for layer_report in report.layer_reports
    )
    if not layer_rows:
        layer_rows = '<tr><td colspan="9">No chain layers found.</td></tr>'

    probe_cards = "\n".join(
        _render_chain_probe_card(probe, index)
        for index, probe in enumerate(_chain_probe_samples(report), 1)
    )
    if not probe_cards:
        probe_cards = '<p class="empty">No chain probes available.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
      --accent: #16a34a;
      --accent-soft: #dcfce7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }}
    main {{
      display: grid;
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .stat, section, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .stat {{
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.25rem;
      letter-spacing: 0;
    }}
    section {{
      padding: 14px;
      overflow-x: auto;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      min-width: 760px;
    }}
    th, td {{
      padding: 7px 8px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      color: #334155;
      font-weight: 700;
      border-top: 0;
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    article {{
      padding: 12px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 0.95rem;
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="subtitle">For each relation layer, the analyzer follows same-layer chains as an induced axis and compares them with non-axis side branches from the same local path. Baseline values shuffle layer colors while preserving graph topology.</p>
  </header>
  <main>
    <div class="summary">
      {_stat("edges", str(report.total_edges))}
      {_stat("strongest axis", escape(strongest_text))}
      {_stat("max depth", str(report.parameters.get("max_depth", "")))}
      {_stat("shuffles", str(report.parameters.get("shuffle_iterations", "")))}
    </div>
    <section>
      <h2>Layer Axes</h2>
      <table>
        <tr><th>layer</th><th>edges</th><th>probes</th><th>axis</th><th>side</th><th>contrast</th><th>ratio</th><th>baseline</th><th>z / p</th></tr>
        {layer_rows}
      </table>
    </section>
    <section>
      <h2>Strongest Local Probes</h2>
      <div class="samples">{probe_cards}</div>
    </section>
  </main>
</body>
</html>
"""


def render_cross_project_alignment_html(
    report: CrossProjectAlignmentReport,
    *,
    title: str = "Chicory Cross-Project Alignment",
) -> str:
    """Render a two-DB relation-layer alignment report."""
    strongest_neighborhood = report.strongest_neighborhood_pair
    strongest_exact = report.strongest_exact_pair
    neighborhood_text = (
        f"{strongest_neighborhood[0]} -> {strongest_neighborhood[1]}"
        f" ({strongest_neighborhood[2]:.2f})"
        if strongest_neighborhood
        else "none"
    )
    exact_text = (
        f"{strongest_exact[0]} -> {strongest_exact[1]}"
        f" ({strongest_exact[2]:.2f})"
        if strongest_exact
        else "none"
    )
    neighborhood_matrix = _render_alignment_matrix(
        report.neighborhood_matrix,
        report.project_a,
        report.project_b,
    )
    exact_matrix = _render_alignment_matrix(
        report.exact_pair_matrix,
        report.project_a,
        report.project_b,
    )
    cell_cards = "\n".join(
        _render_alignment_cell(cell, index)
        for index, cell in enumerate(report.top_cells, 1)
    )
    if not cell_cards:
        cell_cards = '<p class="empty">No shared-anchor cells found.</p>'
    exact_cards = "\n".join(
        _render_alignment_cell(cell, index)
        for index, cell in enumerate(report.top_exact_cells, 1)
    )
    if not exact_cards:
        exact_cards = '<p class="empty">No exact shared edge-pair cells found.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
      --accent: #7c3aed;
      --accent-soft: #ede9fe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }}
    main {{
      display: grid;
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 12px;
    }}
    .stat, section, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .stat {{
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.05rem;
      letter-spacing: 0;
    }}
    section {{
      padding: 14px;
      overflow-x: auto;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    table.matrix {{
      border-collapse: collapse;
      font-size: 0.86rem;
      min-width: 680px;
    }}
    .matrix th, .matrix td {{
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    .matrix th {{
      color: #334155;
      font-weight: 700;
      background: #f1f5f9;
      text-align: center;
    }}
    .matrix .row-head {{
      text-align: left;
      background: #f8fafc;
      color: #334155;
      font-weight: 700;
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    article {{
      padding: 12px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 0.95rem;
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 4px 10px;
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    dt {{ font-weight: 700; color: #334155; }}
    dd {{ margin: 0; }}
    .empty {{
      margin: 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="subtitle">Shared tags act as vertical anchors between two DB planes. Neighborhood cells compare relation-layer strength around the same anchor; exact cells require the same tag pair in both projects.</p>
  </header>
  <main>
    <div class="summary">
      {_stat("project A", escape(report.project_a))}
      {_stat("project B", escape(report.project_b))}
      {_stat("shared anchors", str(report.shared_tag_count))}
      {_stat("exact edge pairs", str(report.exact_pair_count))}
      {_stat("A edges", str(report.edge_count_a))}
      {_stat("B edges", str(report.edge_count_b))}
      {_stat("strongest neighborhood", escape(neighborhood_text))}
      {_stat("strongest exact", escape(exact_text))}
    </div>
    <section>
      <h2>Neighborhood Alignment Matrix</h2>
      {neighborhood_matrix}
    </section>
    <section>
      <h2>Exact Shared Edge-Pair Matrix</h2>
      {exact_matrix}
    </section>
    <section>
      <h2>Top Middle Cells</h2>
      <div class="samples">{cell_cards}</div>
    </section>
    <section>
      <h2>Top Exact Edge Cells</h2>
      <div class="samples">{exact_cards}</div>
    </section>
  </main>
</body>
</html>
"""


def render_hidden_bridges_html(
    report: HiddenBridgeReport,
    *,
    title: str = "Chicory Hidden Bridge Report",
) -> str:
    """Render raw-only cross-project bridge cells."""
    visible = report.visible_summary
    raw = report.raw_summary
    hidden_rows = "\n".join(
        _render_hidden_bridge_row(cell, index)
        for index, cell in enumerate(report.hidden_cells, 1)
    )
    if not hidden_rows:
        hidden_rows = '<tr><td colspan="8">No hidden bridge cells found.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --line: #d7dee8;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.25rem, 2vw, 1.8rem);
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
    }}
    main {{
      display: grid;
      gap: 18px;
      padding: 22px clamp(18px, 4vw, 48px) 36px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 12px;
    }}
    .stat, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .stat {{
      padding: 12px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.05rem;
      letter-spacing: 0;
    }}
    section {{
      padding: 14px;
      overflow-x: auto;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      border-top: 0;
      color: #334155;
      font-weight: 700;
      background: #f1f5f9;
      white-space: nowrap;
    }}
    .score {{
      text-align: right;
      white-space: nowrap;
    }}
    .reason {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      white-space: nowrap;
    }}
    .detail {{
      color: var(--muted);
      max-width: 360px;
      white-space: normal;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="subtitle">Hidden bridges are raw-derived cross-project cells absent from the stricter visible alignment. They are weak-signal candidates, not automatic truths.</p>
  </header>
  <main>
    <div class="summary">
      {_stat("project A", escape(report.project_a))}
      {_stat("project B", escape(report.project_b))}
      {_stat("hidden candidates", str(report.parameters.get("hidden_candidate_count", 0)))}
      {_stat("shown", str(len(report.hidden_cells)))}
      {_stat("visible anchors", str(visible.get("shared_tag_count", 0)))}
      {_stat("raw anchors", str(raw.get("shared_tag_count", 0)))}
      {_stat("visible exact pairs", str(visible.get("exact_pair_count", 0)))}
      {_stat("raw exact pairs", str(raw.get("exact_pair_count", 0)))}
    </div>
    <section>
      <h2>Raw-Only Bridge Cells</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Anchor</th>
            <th>Layers</th>
            <th>Type</th>
            <th>Reason</th>
            <th>Score</th>
            <th>{escape(report.project_a)} detail</th>
            <th>{escape(report.project_b)} detail</th>
          </tr>
        </thead>
        <tbody>{hidden_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _render_alignment_matrix(
    values: tuple[tuple[str, str, float], ...],
    project_a: str,
    project_b: str,
) -> str:
    matrix = {(a, b): score for a, b, score in values}
    max_score = max(matrix.values(), default=0.0)
    header = "".join(f"<th>{escape(layer)}</th>" for layer in LAYER_COLORS)
    rows: list[str] = []
    for layer_a in LAYER_COLORS:
        cells = []
        for layer_b in LAYER_COLORS:
            score = matrix.get((layer_a, layer_b), 0.0)
            alpha = min(0.82, score / max_score * 0.72) if max_score else 0.0
            style = f"background: rgba(124, 58, 237, {alpha:.3f})"
            cells.append(f'<td style="{style}">{score:.2f}</td>')
        rows.append(
            f'<tr><td class="row-head">{escape(layer_a)}</td>{"".join(cells)}</tr>'
        )
    return (
        f'<table class="matrix"><tr><th>{escape(project_a)} \\ {escape(project_b)}</th>'
        f"{header}</tr>{''.join(rows)}</table>"
    )


def _render_alignment_cell(cell: AlignmentCell, index: int) -> str:
    detail_a = "; ".join(cell.detail_a) or "none"
    detail_b = "; ".join(cell.detail_b) or "none"
    return f"""<article>
  <h3>{index}. {escape(cell.anchor)}</h3>
  <dl>
    <dt>type</dt><dd>{escape(cell.cell_type)}</dd>
    <dt>layers</dt><dd>{escape(cell.layer_a)} -> {escape(cell.layer_b)}</dd>
    <dt>score</dt><dd>{cell.score:.2f}</dd>
    <dt>A strength</dt><dd>{cell.score_a:.2f}</dd>
    <dt>B strength</dt><dd>{cell.score_b:.2f}</dd>
    <dt>A detail</dt><dd>{escape(detail_a)}</dd>
    <dt>B detail</dt><dd>{escape(detail_b)}</dd>
  </dl>
</article>"""


def _render_hidden_bridge_row(cell: dict, index: int) -> str:
    detail_a = _detail_list(cell.get("detail_a"))
    detail_b = _detail_list(cell.get("detail_b"))
    return f"""<tr>
  <td>{index}</td>
  <td>{escape(str(cell.get("anchor", "")))}</td>
  <td>{escape(str(cell.get("layer_a", "")))} -> {escape(str(cell.get("layer_b", "")))}</td>
  <td>{escape(str(cell.get("scope", cell.get("cell_type", ""))))}</td>
  <td><span class="reason">{escape(str(cell.get("hidden_reason", "")))}</span></td>
  <td class="score">{float(cell.get("score", 0.0)):.2f}<br><small>{float(cell.get("hidden_score", 0.0)):.2f}</small></td>
  <td class="detail">{escape(detail_a)}</td>
  <td class="detail">{escape(detail_b)}</td>
</tr>"""


def _detail_list(values: object) -> str:
    if isinstance(values, list):
        return "; ".join(str(value) for value in values[:6])
    if isinstance(values, tuple):
        return "; ".join(str(value) for value in values[:6])
    return str(values or "")


def _chain_layer_row(report) -> str:
    baseline = (
        f"{report.baseline_contrast_mean:.2f} +/- "
        f"{report.baseline_contrast_std:.2f}"
    )
    return f"""<tr>
  <td>{escape(report.layer)}</td>
  <td>{report.edge_count}</td>
  <td>{report.probe_count}</td>
  <td>{report.axis_mean:.2f}</td>
  <td>{report.side_mean:.2f}</td>
  <td>{report.contrast:.2f}</td>
  <td>{report.ratio:.2f}</td>
  <td>{escape(baseline)}</td>
  <td>{report.z_score:.2f} / {report.p_value:.3f}</td>
</tr>"""


def _chain_probe_samples(report: ChainAnisotropyReport) -> list[ChainProbe]:
    probes: list[ChainProbe] = []
    for layer_report in report.layer_reports[:4]:
        probes.extend(layer_report.top_probes[:3])
    probes.sort(key=lambda probe: probe.contrast, reverse=True)
    return probes[:12]


def _render_chain_probe_card(probe: ChainProbe, index: int) -> str:
    side_layers = ", ".join(probe.side_layers) or "none"
    return f"""<article>
  <h3>{index}. {escape(probe.axis_layer)}: {escape(probe.start_tag)} -> {escape(probe.next_tag)}</h3>
  <dl>
    <dt>axis length</dt><dd>{probe.axis_length}</dd>
    <dt>side length</dt><dd>{probe.side_length:.2f}</dd>
    <dt>contrast</dt><dd>{probe.contrast:.2f}</dd>
    <dt>side layers</dt><dd>{escape(side_layers)}</dd>
  </dl>
</article>"""


def _stat(label: str, value: str) -> str:
    return f'<div class="stat"><span>{escape(label)}</span><strong>{value}</strong></div>'


def _metric_bar_row(label: str, count: int, total: int) -> str:
    share = count / total if total else 0.0
    width = max(2.0, share * 100.0) if count else 0.0
    return (
        f"<tr><td>{escape(label)}</td><td>{count}</td>"
        f'<td><div class="bar"><span style="width:{width:.1f}%"></span></div> '
        f"{share:.1%}</td></tr>"
    )


def _render_zigzag_sample_card(sample: ZigZagSample, index: int) -> str:
    sources = "; ".join(
        f"{summary} ({count})"
        for summary, count in sample.source_summaries[:2]
    ) or "none"
    return f"""<article>
  <h3>{index}. {escape(' -> '.join(sample.tags))}</h3>
  <dl>
    <dt>layers</dt><dd>{escape(' / '.join(sample.side_layers))}</dd>
    <dt>class</dt><dd>{escape(sample.orientation_class)}</dd>
    <dt>signature</dt><dd>{escape(sample.signature)}</dd>
    <dt>vector</dt><dd>{sample.vector_score:.3f}</dd>
    <dt>center</dt><dd>{sample.center_score:.2f}</dd>
    <dt>void</dt><dd>{sample.void_score:.2f}</dd>
    <dt>diagonals</dt><dd>{escape(sample.ac_status)} / {escape(sample.bd_status)}</dd>
    <dt>score</dt><dd>{sample.interestingness:.2f}</dd>
    <dt>sources</dt><dd>{escape(sources)}</dd>
  </dl>
</article>"""


def _render_motif_card(motif: SquareMotif, index: int) -> str:
    svg = _render_motif_svg(motif)
    repeated = ", ".join(motif.repeated_color_vertices) or "none"
    sources = "; ".join(
        f"{summary} ({count})"
        for summary, count in motif.source_summaries[:3]
    ) or "none"
    return f"""<article>
  <h2>{index}. {escape(' -> '.join(motif.tags))}</h2>
  {svg}
  <dl>
    <dt>layers</dt><dd>{escape(' / '.join(motif.side_layers))}</dd>
    <dt>AC</dt><dd>{escape(_diagonal_summary(motif.ac_diagonal))}</dd>
    <dt>BD</dt><dd>{escape(_diagonal_summary(motif.bd_diagonal))}</dd>
    <dt>center</dt><dd>{motif.center_score:.2f}</dd>
    <dt>void</dt><dd>{motif.void_score:.2f}</dd>
    <dt>score</dt><dd>{motif.interestingness:.2f}</dd>
    <dt>repeats</dt><dd>{escape(repeated)}</dd>
    <dt>sources</dt><dd>{escape(sources)}</dd>
  </dl>
</article>"""


def _render_vertical_motif_card(
    motif: VerticalSquareMotif,
    index: int,
) -> str:
    svg = _render_vertical_motif_svg(motif)
    layer = motif.relation_layer or "none"
    return f"""<article>
  <h2>{index}. {escape(motif.sources[0])} x {escape(motif.sources[1])}</h2>
  {svg}
  <dl>
    <dt>columns</dt><dd>{escape(' / '.join(motif.tags))}</dd>
    <dt>relation</dt><dd>{escape(layer)} ({escape(motif.relation_status)}), score={motif.relation_score:.2f}</dd>
    <dt>counts</dt><dd>{motif.counts[0]}, {motif.counts[1]}, {motif.counts[2]}, {motif.counts[3]}</dd>
    <dt>balance</dt><dd>{motif.source_balance:.2f}</dd>
    <dt>score</dt><dd>{motif.interestingness:.2f}</dd>
  </dl>
</article>"""


def _render_tensor_vertical_motif_card(
    motif: TensorVerticalSquareMotif,
    index: int,
) -> str:
    svg = _render_tensor_vertical_motif_svg(motif)
    column_layer = motif.column_relation_layer or "none"
    row_layer = motif.row_relation_layer or "none"
    orientation = (
        f"{motif.orientation_class}; diagonal={motif.diagonal_bias}; "
        f"variants={motif.symmetry_variants}"
    )
    return f"""<article>
  <h2>{index}. {escape(motif.sources[0])} x {escape(motif.sources[1])}</h2>
  {svg}
  <dl>
    <dt>rows</dt><dd>{escape(' / '.join(motif.row_tags))}</dd>
    <dt>columns</dt><dd>{escape(' / '.join(motif.column_tags))}</dd>
    <dt>cell layers</dt><dd>{escape(' / '.join(motif.cell_layers))}</dd>
    <dt>cell scores</dt><dd>{', '.join(f'{score:.2f}' for score in motif.cell_scores)}</dd>
    <dt>orientation</dt><dd>{escape(orientation)}</dd>
    <dt>signature</dt><dd>{escape(motif.color_signature)}</dd>
    <dt>column relation</dt><dd>{escape(column_layer)} ({escape(motif.column_relation_status)}), score={motif.column_relation_score:.2f}</dd>
    <dt>row relation</dt><dd>{escape(row_layer)}, score={motif.row_relation_score:.2f}</dd>
    <dt>score</dt><dd>{motif.interestingness:.2f}</dd>
  </dl>
</article>"""


def _render_tensor_vertical_motif_svg(
    motif: TensorVerticalSquareMotif,
) -> str:
    x1, x2 = 178, 326
    y1, y2 = 130, 260
    column_color = LAYER_COLORS.get(motif.column_relation_layer or "", "#94a3b8")
    relation_class = (
        "column-relation"
        if motif.column_relation_layer
        else "column-relation column-relation-missing"
    )
    return f"""<svg viewBox="0 0 420 340" role="img" aria-label="{escape('tensor vertical square motif')}">
  <text class="label tag-label" x="{x1}" y="54">{escape(_short_label(motif.column_tags[0], 18))}</text>
  <text class="label tag-label" x="{x2}" y="54">{escape(_short_label(motif.column_tags[1], 18))}</text>
  <line class="{relation_class}" x1="{x1}" y1="76" x2="{x2}" y2="76" stroke="{column_color}" stroke-width="{_relation_width(motif.column_relation_score):.1f}">
    <title>{escape(motif.column_relation_status)} column relation</title>
  </line>
  <text class="label source-label" x="138" y="{y1}">{escape(_short_label(motif.sources[0], 24))}</text>
  <text class="label source-label" x="138" y="{y2}">{escape(_short_label(motif.sources[1], 24))}</text>
  {_tensor_cell_line(x1, y1, motif.cell_layers[0], motif.cell_scores[0], motif.row_tags[0], motif.column_tags[0])}
  {_tensor_cell_line(x2, y1, motif.cell_layers[1], motif.cell_scores[1], motif.row_tags[0], motif.column_tags[1])}
  {_tensor_cell_line(x1, y2, motif.cell_layers[2], motif.cell_scores[2], motif.row_tags[1], motif.column_tags[0])}
  {_tensor_cell_line(x2, y2, motif.cell_layers[3], motif.cell_scores[3], motif.row_tags[1], motif.column_tags[1])}
</svg>"""


def _tensor_cell_line(
    x: int,
    y: int,
    layer: str,
    score: float,
    row_tag: str,
    column_tag: str,
) -> str:
    color = LAYER_COLORS.get(layer, "#334155")
    width = _edge_width(score)
    return f"""<g>
  <line class="cell-line" x1="{x - 38}" y1="{y}" x2="{x + 38}" y2="{y}" stroke="{color}" stroke-width="{width:.1f}">
    <title>{escape(row_tag)} - {escape(column_tag)}: {escape(layer)} {score:.2f}</title>
  </line>
  <text class="label tag-label" x="{x}" y="{y + 24}">{score:.1f}</text>
</g>"""


def _render_vertical_motif_svg(motif: VerticalSquareMotif) -> str:
    x1, x2 = 178, 326
    y1, y2 = 130, 260
    color = LAYER_COLORS.get(motif.relation_layer or "", "#94a3b8")
    relation_class = "relation" if motif.relation_layer else "relation relation-missing"
    relation_label = (
        f"{motif.tags[0]} - {motif.tags[1]}: "
        f"{motif.relation_layer or 'no tensor edge'} {motif.relation_score:.2f}"
    )
    counts = motif.counts
    return f"""<svg viewBox="0 0 420 340" role="img" aria-label="{escape('vertical square motif')}">
  <text class="label tag-label" x="{x1}" y="54">{escape(_short_label(motif.tags[0], 18))}</text>
  <text class="label tag-label" x="{x2}" y="54">{escape(_short_label(motif.tags[1], 18))}</text>
  <line class="{relation_class}" x1="{x1}" y1="76" x2="{x2}" y2="76" stroke="{color}" stroke-width="{_relation_width(motif.relation_score):.1f}">
    <title>{escape(relation_label)}</title>
  </line>
  <text class="label source-label" x="138" y="{y1}">{escape(_short_label(motif.sources[0], 24))}</text>
  <text class="label source-label" x="138" y="{y2}">{escape(_short_label(motif.sources[1], 24))}</text>
  <line class="grid-line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y1}"></line>
  <line class="grid-line" x1="{x1}" y1="{y2}" x2="{x2}" y2="{y2}"></line>
  <line class="grid-line" x1="{x1}" y1="{y1}" x2="{x1}" y2="{y2}"></line>
  <line class="grid-line" x1="{x2}" y1="{y1}" x2="{x2}" y2="{y2}"></line>
  {_vertical_cell(x1, y1, counts[0], motif.sources[0], motif.tags[0])}
  {_vertical_cell(x2, y1, counts[1], motif.sources[0], motif.tags[1])}
  {_vertical_cell(x1, y2, counts[2], motif.sources[1], motif.tags[0])}
  {_vertical_cell(x2, y2, counts[3], motif.sources[1], motif.tags[1])}
</svg>"""


def _vertical_cell(x: int, y: int, count: int, source: str, tag: str) -> str:
    radius = min(24, 9 + 4 * math.log1p(count))
    return f"""<g transform="translate({x} {y})">
  <circle class="cell" r="{radius:.1f}">
    <title>{escape(source)} has tag {escape(tag)} on {count} memory chunk(s)</title>
  </circle>
  <text class="label tag-label" x="0" y="0">{count}</text>
</g>"""


def _render_motif_svg(motif: SquareMotif) -> str:
    points = ((78, 70), (302, 70), (302, 294), (78, 294))
    sides = (
        (0, 1, motif.side_edges[0]),
        (1, 2, motif.side_edges[1]),
        (2, 3, motif.side_edges[2]),
        (3, 0, motif.side_edges[3]),
    )
    side_lines = "\n".join(
        _line(
            points[a],
            points[b],
            LAYER_COLORS.get(edge.dominant_layer, "#334155"),
            width=_edge_width(edge.score),
            cls="side",
            title=f"{edge.tag_a} - {edge.tag_b}: {edge.dominant_layer} {edge.score:.2f}",
        )
        for a, b, edge in sides
    )
    diagonals = "\n".join(
        (
            _diagonal(points[0], points[2], motif.ac_diagonal),
            _diagonal(points[1], points[3], motif.bd_diagonal),
        )
    )
    nodes = "\n".join(
        _node(point, tag)
        for point, tag in zip(points, motif.tags)
    )
    center_radius = 7 + 16 * max(0.0, min(1.0, motif.center_score))
    center = f"""
    <circle class="center-dot" cx="190" cy="182" r="{center_radius:.1f}">
      <title>center score {motif.center_score:.2f}</title>
    </circle>
    <text class="center-label" x="190" y="182">{motif.center_score:.2f}</text>
"""
    return f"""<svg viewBox="0 0 380 364" role="img" aria-label="{escape('square motif')}">
  {diagonals}
  {side_lines}
  {center}
  {nodes}
</svg>"""


def _line(
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    *,
    width: float,
    cls: str,
    title: str,
) -> str:
    return (
        f'<line class="{cls}" x1="{start[0]}" y1="{start[1]}" '
        f'x2="{end[0]}" y2="{end[1]}" stroke="{color}" '
        f'stroke-width="{width:.1f}"><title>{escape(title)}</title></line>'
    )


def _diagonal(
    start: tuple[int, int],
    end: tuple[int, int],
    signal: DiagonalSignal,
) -> str:
    color = DIAGONAL_COLORS.get(signal.status, "#94a3b8")
    return _line(
        start,
        end,
        color,
        width=2.4,
        cls=f"diagonal diag-{signal.status}",
        title=_diagonal_summary(signal),
    )


def _node(point: tuple[int, int], label: str) -> str:
    short = _short_label(label)
    return f"""<g class="node" transform="translate({point[0]} {point[1]})">
  <circle r="34"><title>{escape(label)}</title></circle>
  <text>{escape(short)}</text>
</g>"""


def _edge_width(score: float) -> float:
    return max(2.4, min(9.0, 2.0 + score * 0.9))


def _relation_width(score: float) -> float:
    return max(2.4, min(9.0, 2.4 + score * 0.7))


def _short_label(label: str, max_len: int = 16) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "..."


def _diagonal_summary(signal: DiagonalSignal) -> str:
    endpoints = "-".join(signal.endpoints)
    return (
        f"{endpoints} {signal.status}; "
        f"support={signal.support:.2f}; gap={signal.gap:.2f}; "
        f"consistency={signal.path_consistency:.2f}"
    )
