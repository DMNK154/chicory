"""Dash web dashboard for Chicory memory system."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

from chicory.dashboard import data

# ── Styling constants ────────────────────────────────────────────────

BG = "#111111"
CARD_BG = "#1e1e1e"
CARD_BORDER = "#333333"
TEXT = "#e0e0e0"
MUTED = "#888888"
ACCENT = "#4fc3f7"

QUADRANT_COLORS = {
    "active_deep_work": {"marker": "#4caf50", "bg": "rgba(76,175,80,0.08)"},
    "novel_exploration": {"marker": "#ffeb3b", "bg": "rgba(255,235,59,0.08)"},
    "dormant_reactivation": {"marker": "#f44336", "bg": "rgba(244,67,54,0.08)"},
    "inactive": {"marker": "#9e9e9e", "bg": "rgba(158,158,158,0.08)"},
}

SYNC_COLORS = {
    "low_trend_high_retrieval": "#42a5f5",
    "cross_domain_bridge": "#ffa726",
    "unexpected_semantic_cluster": "#ab47bc",
}

META_COLORS = {
    "recurring_sync": "#42a5f5",
    "cross_domain_theme": "#ffa726",
    "emergent_category": "#ab47bc",
}

TAB_STYLE = {
    "backgroundColor": "#1a1a1a",
    "color": MUTED,
    "borderBottom": "2px solid #333",
    "padding": "12px 20px",
}
TAB_SELECTED = {
    "backgroundColor": CARD_BG,
    "color": ACCENT,
    "borderBottom": f"2px solid {ACCENT}",
    "padding": "12px 20px",
}

CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {CARD_BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "margin": "8px",
    "flex": "1",
}


def _empty_fig(message: str = "No data yet") -> go.Figure:
    """Return a dark empty figure with a centered message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        annotations=[
            dict(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=MUTED),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Tab renderers ────────────────────────────────────────────────────


def _render_overview(db_path: Path) -> html.Div:
    d = data.get_overview(db_path)
    trends = d["trends"]

    cards = html.Div(
        [
            _stat_card("Memories", d["memory_count"]),
            _stat_card("Active Tags", d["tag_count"]),
            _stat_card(
                "Hot Tags",
                sum(1 for t in trends if t["temperature"] > 0.7),
            ),
            _stat_card("Sync Events", d["sync_count"]),
            _stat_card("Meta-Patterns", d["meta_count"]),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "20px"},
    )

    if trends:
        names = [t["tag"] for t in trends]
        temps = [t["temperature"] for t in trends]
        fig = go.Figure(
            go.Heatmap(
                z=[temps],
                x=names,
                y=[""],
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=1,
                hovertemplate="%{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="Temp"),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            title="Tag Temperature",
            height=180 + max(0, len(names) - 15) * 5,
            margin=dict(l=40, r=40, t=50, b=80),
            xaxis=dict(tickangle=-45),
        )
    else:
        fig = _empty_fig("No tag trends recorded yet")

    return html.Div([cards, dcc.Graph(figure=fig, config={"displayModeBar": False})])


def _stat_card(label: str, value: Any) -> html.Div:
    return html.Div(
        [
            html.Div(str(value), style={"fontSize": "36px", "fontWeight": "bold", "color": ACCENT}),
            html.Div(label, style={"fontSize": "14px", "color": MUTED, "marginTop": "4px"}),
        ],
        style=CARD_STYLE,
    )


def _render_phase_space(db_path: Path) -> html.Div:
    ps = data.get_phase_space(db_path)["phase_space"]

    fig = go.Figure()

    # Quadrant background regions
    for qname, coords in [
        ("inactive", (0, 0, 0.5, 0.5)),
        ("novel_exploration", (0.5, 0, 1, 0.5)),
        ("dormant_reactivation", (0, 0.5, 0.5, 1)),
        ("active_deep_work", (0.5, 0.5, 1, 1)),
    ]:
        x0, y0, x1, y1 = coords
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=QUADRANT_COLORS[qname]["bg"],
            line=dict(width=0),
            layer="below",
        )

    # Quadrant labels
    labels = {
        "active_deep_work": (0.75, 0.75),
        "novel_exploration": (0.75, 0.25),
        "dormant_reactivation": (0.25, 0.75),
        "inactive": (0.25, 0.25),
    }
    pretty = {
        "active_deep_work": "Active Deep Work",
        "novel_exploration": "Novel Exploration",
        "dormant_reactivation": "Dormant Reactivation",
        "inactive": "Inactive",
    }
    for qname, (lx, ly) in labels.items():
        fig.add_annotation(
            x=lx, y=ly, text=pretty[qname],
            showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.15)"),
        )

    has_points = False
    for qname, items in ps.items():
        if not items:
            continue
        has_points = True
        fig.add_trace(go.Scatter(
            x=[p["temperature"] for p in items],
            y=[p["retrieval_freq"] for p in items],
            mode="markers+text",
            name=pretty.get(qname, qname),
            text=[p["tag"] for p in items],
            textposition="top center",
            textfont=dict(size=9, color=TEXT),
            marker=dict(
                size=[max(8, min(20, abs(p["off_diagonal"]) * 30 + 8)) for p in items],
                color=QUADRANT_COLORS.get(qname, QUADRANT_COLORS["inactive"])["marker"],
                opacity=0.85,
            ),
            hovertemplate="%{text}<br>Temp: %{x:.2f}<br>Retrieval: %{y:.2f}<extra></extra>",
        ))

    if not has_points:
        fig = _empty_fig("No phase space data yet")
    else:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            title="Phase Space",
            xaxis=dict(title="Temperature", range=[0, 1]),
            yaxis=dict(title="Retrieval Frequency", range=[0, 1]),
            height=550,
            margin=dict(l=60, r=40, t=50, b=60),
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

    return html.Div([dcc.Graph(figure=fig)])


def _render_trends(db_path: Path) -> html.Div:
    overview = data.get_overview(db_path)
    tag_options = [{"label": n, "value": n} for n in overview["tag_names"]]
    default = tag_options[0]["value"] if tag_options else None

    return html.Div([
        dcc.Dropdown(
            id="trend-tag-select",
            options=tag_options,
            value=default,
            placeholder="Select a tag...",
            style={"backgroundColor": CARD_BG, "color": TEXT, "marginBottom": "16px"},
        ),
        dcc.Graph(id="trend-graph", figure=_empty_fig("Select a tag to view trends")),
    ])


def _build_trend_figure(history: list[dict]) -> go.Figure:
    if not history:
        return _empty_fig("No trend snapshots recorded for this tag yet")

    dates = [h["computed_at"] for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=[h["temperature"] for h in history],
        name="Temperature", mode="lines",
        fill="tozeroy", fillcolor="rgba(255,87,34,0.15)",
        line=dict(color="#ff5722", width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[h["level"] for h in history],
        name="Level", mode="lines",
        line=dict(color="#42a5f5", width=1.5),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[h["velocity"] for h in history],
        name="Velocity", mode="lines",
        line=dict(color="#66bb6a", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[h["jerk"] for h in history],
        name="Jerk", mode="lines",
        line=dict(color="#fdd835", width=1, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title="Trend History",
        height=450,
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(title="Time"),
        yaxis=dict(title="Temperature", range=[0, 1], side="left"),
        yaxis2=dict(
            title="Level / Velocity / Jerk",
            overlaying="y", side="right",
        ),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _render_sync(db_path: Path) -> html.Div:
    result = data.get_synchronicities(db_path)
    events = result["synchronicities"]
    velocity = result.get("velocity", {})

    if not events:
        return html.Div([
            dcc.Graph(figure=_empty_fig("No synchronicity events detected yet")),
        ])

    # Timeline scatter
    timeline = go.Figure()
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)

    for etype, items in by_type.items():
        timeline.add_trace(go.Scatter(
            x=[e["detected_at"] for e in items],
            y=[e["effective_strength"] for e in items],
            mode="markers",
            name=etype.replace("_", " ").title(),
            marker=dict(
                size=[max(6, min(18, e["effective_strength"] * 5)) for e in items],
                color=SYNC_COLORS.get(etype, MUTED),
                opacity=0.85,
            ),
            hovertemplate=(
                "%{text}<br>Strength: %{y:.2f}<br>%{x}<extra></extra>"
            ),
            text=[e["description"][:80] for e in items],
        ))

    timeline.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title="Synchronicity Timeline",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Effective Strength"),
        height=400,
        margin=dict(l=60, r=40, t=50, b=60),
        legend=dict(x=0.01, y=0.99),
    )

    # Type distribution pie
    type_counts = {t: len(items) for t, items in by_type.items()}
    pie = go.Figure(go.Pie(
        labels=[t.replace("_", " ").title() for t in type_counts],
        values=list(type_counts.values()),
        marker=dict(colors=[SYNC_COLORS.get(t, MUTED) for t in type_counts]),
        textinfo="label+value",
        textfont=dict(color=TEXT),
        hole=0.4,
    ))
    pie.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title="Event Types",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    # Velocity cards
    vel_cards = html.Div(
        [
            _stat_card("Sync Level", f"{velocity.get('level', 0):.2f}"),
            _stat_card("Sync Velocity", f"{velocity.get('velocity', 0):.2f}"),
            _stat_card("Sync Jerk", f"{velocity.get('jerk', 0):.2f}"),
            _stat_card("Event Count", velocity.get("event_count", 0)),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginTop": "12px"},
    )

    return html.Div([
        html.Div(
            [
                html.Div([dcc.Graph(figure=timeline)], style={"flex": "2"}),
                html.Div([dcc.Graph(figure=pie)], style={"flex": "1"}),
            ],
            style={"display": "flex", "gap": "12px"},
        ),
        vel_cards,
    ])


def _render_lattice(db_path: Path) -> html.Div:
    state = data.get_lattice(db_path)
    positions = state.get("positions", [])
    resonances = state.get("resonances", [])
    void_profile = state.get("void_profile")

    if not positions:
        return html.Div([
            dcc.Graph(figure=_empty_fig("No lattice positions yet")),
        ])

    # Polar scatter
    fig = go.Figure()

    # Group by event type
    by_type: dict[str, list[dict]] = {}
    for p in positions:
        by_type.setdefault(p["event_type"], []).append(p)

    for etype, items in by_type.items():
        fig.add_trace(go.Scatterpolar(
            r=[p["event_strength"] for p in items],
            theta=[p["angle"] * 180 / math.pi for p in items],
            mode="markers",
            name=etype.replace("_", " ").title(),
            marker=dict(
                size=[max(6, min(16, p["event_strength"] * 6)) for p in items],
                color=SYNC_COLORS.get(etype, MUTED),
                opacity=0.85,
            ),
            hovertemplate=(
                "%{text}<br>Strength: %{r:.2f}<br>Angle: %{theta:.1f}<extra></extra>"
            ),
            text=[p["event_description"][:60] for p in items],
        ))

    # Resonance chords (top 10 by strength)
    pos_lookup = {p["sync_event_id"]: p for p in positions}
    for res in sorted(resonances, key=lambda r: r["strength"], reverse=True)[:10]:
        eids = res["event_ids"]
        if len(eids) >= 2 and eids[0] in pos_lookup and eids[1] in pos_lookup:
            p1 = pos_lookup[eids[0]]
            p2 = pos_lookup[eids[1]]
            fig.add_trace(go.Scatterpolar(
                r=[p1["event_strength"], p2["event_strength"]],
                theta=[p1["angle"] * 180 / math.pi, p2["angle"] * 180 / math.pi],
                mode="lines",
                line=dict(
                    color=f"rgba(79,195,247,{min(0.8, res['strength'] / 10)})",
                    width=max(1, min(3, res["strength"] / 3)),
                ),
                showlegend=False,
                hovertemplate=f"Resonance: {len(res['shared_primes'])} primes<br>Strength: {res['strength']:.2f}<extra></extra>",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title="Prime Ramsey Lattice",
        height=550,
        margin=dict(l=40, r=40, t=50, b=40),
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        legend=dict(x=0.01, y=0.99),
    )

    children: list = [dcc.Graph(figure=fig)]

    # Void profile card
    if void_profile:
        children.append(html.Div(
            [
                html.H4("Void Profile", style={"color": ACCENT, "marginBottom": "8px"}),
                html.P(void_profile["description"], style={"color": TEXT}),
                html.P(
                    f"Void radius: {void_profile['void_radius']:.4f} | "
                    f"Edge tags: {', '.join(void_profile['edge_tags'])}",
                    style={"color": MUTED, "fontSize": "13px"},
                ),
            ],
            style={**CARD_STYLE, "marginTop": "12px"},
        ))

    # Resonance table
    if resonances:
        table_data = [
            {
                "Event IDs": ", ".join(str(e) for e in r["event_ids"]),
                "Shared Primes": len(r["shared_primes"]),
                "Strength": round(r["strength"], 3),
                "Description": r["description"][:100],
            }
            for r in sorted(resonances, key=lambda x: x["strength"], reverse=True)[:20]
        ]
        children.append(
            dash_table.DataTable(
                data=table_data,
                columns=[{"name": c, "id": c} for c in table_data[0]],
                style_header={"backgroundColor": "#222", "color": ACCENT, "fontWeight": "bold"},
                style_cell={
                    "backgroundColor": CARD_BG,
                    "color": TEXT,
                    "border": f"1px solid {CARD_BORDER}",
                    "textAlign": "left",
                    "padding": "8px",
                    "maxWidth": "400px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_table={"marginTop": "16px"},
            )
        )

    return html.Div(children)


def _render_meta(db_path: Path) -> html.Div:
    result = data.get_meta_patterns(db_path)
    patterns = result["meta_patterns"]

    if not patterns:
        return html.Div(
            html.P("No meta-patterns detected yet.", style={"color": MUTED, "padding": "40px", "textAlign": "center"}),
        )

    cards = []
    for p in patterns:
        ptype = p["type"]
        color = META_COLORS.get(ptype, MUTED)
        confidence = p.get("confidence", 0)

        cards.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            ptype.replace("_", " ").title(),
                            style={
                                "backgroundColor": color,
                                "color": "#111",
                                "padding": "3px 10px",
                                "borderRadius": "12px",
                                "fontSize": "12px",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Span(
                            p.get("detected_at", "")[:16] if p.get("detected_at") else "",
                            style={"color": MUTED, "fontSize": "12px", "marginLeft": "12px"},
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.P(p["description"], style={"color": TEXT, "marginBottom": "10px"}),
                # Confidence bar
                html.Div(
                    [
                        html.Div(
                            f"{confidence:.0%}",
                            style={
                                "width": f"{confidence * 100}%",
                                "backgroundColor": color,
                                "color": "#111",
                                "padding": "2px 8px",
                                "borderRadius": "4px",
                                "fontSize": "12px",
                                "fontWeight": "bold",
                                "minWidth": "40px",
                                "textAlign": "center",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": "#333",
                        "borderRadius": "4px",
                        "overflow": "hidden",
                    },
                ),
            ],
            style={**CARD_STYLE, "flex": "none", "marginBottom": "12px"},
        ))

    return html.Div(cards)


def _build_network_figure(net: dict) -> go.Figure:
    """Build a force-directed network graph from node/edge data."""
    nodes = net["nodes"]
    edges = net["edges"]

    if not nodes:
        return _empty_fig("No tags in the system yet")

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["tag"])
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, len(nodes) ** 0.5))

    # Build edge traces (single trace with None gaps)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_weights: list[int] = []
    for e in edges:
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(e["weight"])

    max_weight = max(edge_weights) if edge_weights else 1
    avg_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 1

    fig = go.Figure()

    # Thin edges (below-average weight)
    thin_x: list[float | None] = []
    thin_y: list[float | None] = []
    thick_x: list[float | None] = []
    thick_y: list[float | None] = []
    for i, e in enumerate(edges):
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        if e["weight"] <= avg_weight:
            thin_x.extend([x0, x1, None])
            thin_y.extend([y0, y1, None])
        else:
            thick_x.extend([x0, x1, None])
            thick_y.extend([y0, y1, None])

    if thin_x:
        fig.add_trace(go.Scatter(
            x=thin_x, y=thin_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(255,255,255,0.12)"),
            hoverinfo="skip",
            showlegend=False,
        ))
    if thick_x:
        fig.add_trace(go.Scatter(
            x=thick_x, y=thick_y,
            mode="lines",
            line=dict(width=2, color="rgba(79,195,247,0.3)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Build node traces grouped by quadrant
    max_events = max((n["event_count"] for n in nodes), default=1) or 1
    pretty_q = {
        "active_deep_work": "Active Deep Work",
        "novel_exploration": "Novel Exploration",
        "dormant_reactivation": "Dormant Reactivation",
        "inactive": "Inactive",
    }

    by_quadrant: dict[str, list[dict]] = {}
    for n in nodes:
        by_quadrant.setdefault(n["quadrant"], []).append(n)

    for qname, items in by_quadrant.items():
        color = QUADRANT_COLORS.get(qname, QUADRANT_COLORS["inactive"])["marker"]
        fig.add_trace(go.Scatter(
            x=[pos[n["tag"]][0] for n in items if n["tag"] in pos],
            y=[pos[n["tag"]][1] for n in items if n["tag"] in pos],
            mode="markers+text",
            name=pretty_q.get(qname, qname),
            text=[n["tag"] for n in items if n["tag"] in pos],
            textposition="top center",
            textfont=dict(size=10, color=TEXT),
            marker=dict(
                size=[max(10, min(40, n["temperature"] * 35 + 8)) for n in items if n["tag"] in pos],
                color=color,
                opacity=[max(0.4, min(1.0, 0.4 + 0.6 * n["event_count"] / max_events)) for n in items if n["tag"] in pos],
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate=(
                "%{text}<br>"
                "Temperature: %{customdata[0]:.3f}<br>"
                "Retrieval Freq: %{customdata[1]:.3f}<br>"
                "Events: %{customdata[2]}<br>"
                "Quadrant: %{customdata[3]}"
                "<extra></extra>"
            ),
            customdata=[
                [n["temperature"], n["retrieval_freq"], n["event_count"], pretty_q.get(n["quadrant"], n["quadrant"])]
                for n in items if n["tag"] in pos
            ],
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title="Tag Co-occurrence Network",
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _render_network(db_path: Path) -> html.Div:
    net = data.get_network_data(db_path)
    fig = _build_network_figure(net)

    stats = html.Div(
        [
            _stat_card("Nodes", len(net["nodes"])),
            _stat_card("Edges", len(net["edges"])),
            _stat_card(
                "Components",
                _count_components(net) if net["nodes"] else 0,
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"},
    )

    return html.Div([
        html.Div(
            [
                stats,
                html.Button(
                    "Refresh",
                    id="network-refresh-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": ACCENT,
                        "color": "#111",
                        "border": "none",
                        "borderRadius": "6px",
                        "padding": "8px 20px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "marginBottom": "12px",
                    },
                ),
            ],
        ),
        dcc.Graph(id="network-graph", figure=fig),
    ])


def _count_components(net: dict) -> int:
    """Count connected components in the network."""
    G = nx.Graph()
    for n in net["nodes"]:
        G.add_node(n["tag"])
    for e in net["edges"]:
        G.add_edge(e["source"], e["target"])
    return nx.number_connected_components(G)


# ── App factory ──────────────────────────────────────────────────────


def create_app(db_path: Path) -> Dash:
    """Build and return the Dash application."""
    app = Dash(
        __name__,
        title="Chicory Dashboard",
        update_title="Refreshing...",
    )

    app.layout = html.Div(
        [
            dcc.Store(id="db-path", data=str(db_path)),
            dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
            html.H1(
                "Chicory Memory System",
                style={"textAlign": "center", "color": TEXT, "margin": "16px 0"},
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Phase Space", value="phase", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Trends", value="trends", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Synchronicity", value="sync", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Lattice", value="lattice", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Meta-Patterns", value="meta", style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="Network", value="network", style=TAB_STYLE, selected_style=TAB_SELECTED),
                ],
            ),
            html.Div(id="tab-content", style={"padding": "20px 0"}),
        ],
        style={"backgroundColor": BG, "minHeight": "100vh", "padding": "20px", "fontFamily": "system-ui, sans-serif"},
    )

    _register_callbacks(app)
    return app


def _register_callbacks(app: Dash) -> None:
    """Wire up all Dash callbacks."""

    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "value"), Input("refresh-interval", "n_intervals")],
        [State("db-path", "data")],
    )
    def render_tab(tab: str, _n: int, db_path_str: str) -> html.Div:
        # Skip auto-refresh for the network tab (manual refresh only)
        if ctx.triggered_id == "refresh-interval" and tab == "network":
            from dash import no_update
            return no_update

        db_path = Path(db_path_str)
        renderers = {
            "overview": _render_overview,
            "phase": _render_phase_space,
            "trends": _render_trends,
            "sync": _render_sync,
            "lattice": _render_lattice,
            "meta": _render_meta,
            "network": _render_network,
        }
        renderer = renderers.get(tab, _render_overview)
        return renderer(db_path)

    @app.callback(
        Output("trend-graph", "figure"),
        [Input("trend-tag-select", "value")],
        [State("db-path", "data")],
        prevent_initial_call=False,
    )
    def update_trend_graph(tag_name: str | None, db_path_str: str) -> go.Figure:
        if not tag_name:
            return _empty_fig("Select a tag to view trends")
        history = data.get_trend_history(Path(db_path_str), tag_name)
        return _build_trend_figure(history)

    @app.callback(
        Output("network-graph", "figure"),
        [Input("network-refresh-btn", "n_clicks")],
        [State("db-path", "data")],
        prevent_initial_call=True,
    )
    def refresh_network(_n: int, db_path_str: str) -> go.Figure:
        net = data.get_network_data(Path(db_path_str))
        return _build_network_figure(net)


def run_dashboard(
    db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Entry point: create and run the Dash server."""
    app = create_app(db_path)
    print(f"Chicory dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
