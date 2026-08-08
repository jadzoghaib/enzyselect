"""Plotly figures for EnzySelect.

Every figure follows the same house rules, which are enforced here rather than
repeated at each call site:

* Colour follows the *entity*, never its rank — component colours are fixed in
  ``config.COMPONENT_COLORS`` and never reassigned when a ranking changes.
* One y-axis per chart. There are no dual-axis figures anywhere in this app.
* Hairline solid gridlines, thin marks, generous padding, no chart junk.
* A 2px surface-coloured spacer separates adjacent and stacked fills.
* A legend whenever there are two or more series; direct labels used
  selectively, never a number on every point.
* Every chart has a table-view twin rendered beside it in ``app.py``, so no
  value is reachable only by hovering.

The palette is the validated default palette; the specific slot sets used here
were checked with the palette validator against the light chart surface
``#fcfcfb`` (see the note in ``config.PALETTE``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import (
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    PALETTE,
    PLOTLY_FONT,
)
from .scoring import COMPONENT_KEYS

SPACER = 2  # px of surface colour between adjacent fills
CORNER = 4  # px rounded data-end


def _base_layout(fig: go.Figure, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(family=PLOTLY_FONT, size=13, color=PALETTE["text_secondary"]),
        # Top margin holds the title AND the legend row beneath it; anything
        # tighter and the two collide. Side/bottom margins start small and are
        # grown automatically by the axes (see automargin below), which is what
        # keeps long category labels and axis titles from being clipped.
        margin=dict(l=8, r=8, t=92, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text_secondary"]),
        ),
        title=dict(
            font=dict(size=15, color=PALETTE["text_primary"]),
            x=0, xanchor="left", y=0.97, yanchor="top",
        ),
        hoverlabel=dict(font=dict(family=PLOTLY_FONT, size=12)),
    )
    axis = dict(
        gridcolor=PALETTE["grid"],
        griddash="solid",
        linecolor=PALETTE["axis"],
        zeroline=False,
        # Without automargin, long candidate names on a horizontal bar chart
        # get truncated to a single character against an 8px margin.
        automargin=True,
        tickfont=dict(color=PALETTE["muted"], size=12),
        title=dict(font=dict(color=PALETTE["text_secondary"], size=12)),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# --------------------------------------------------------------------------
# Ranked candidates
# --------------------------------------------------------------------------
def top_candidates_bar(scored: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Top-N overall scores.

    One measure, so one hue. Shortlisted candidates carry the series colour and
    the rest recede to muted grey — emphasis, not a value ramp on nominal
    categories.
    """
    data = scored.head(top_n).iloc[::-1]
    fig = go.Figure()
    for shortlisted, label, color in (
        (True, "Shortlisted for testing", PALETTE["series"][0]),
        (False, "Below the shortlist cut", PALETTE["muted"]),
    ):
        subset = data[data["shortlisted"] == shortlisted]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["overall_score"],
                y=subset["enzyme_name"],
                orientation="h",
                name=label,
                marker=dict(
                    color=color,
                    cornerradius=CORNER,
                    line=dict(color=PALETTE["surface"], width=SPACER),
                ),
                text=[f"{v:.1f}" for v in subset["overall_score"]],
                textposition="outside",
                textfont=dict(color=PALETTE["text_secondary"], size=12),
                hovertemplate="<b>%{y}</b><br>Illustrative score %{x:.1f}/100<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Top {min(top_n, len(scored))} candidates by illustrative score",
        bargap=0.3,
    )
    fig.update_xaxes(title="Illustrative score (0–100)", range=[0, 108])
    return _base_layout(fig, height=max(380, 26 * len(data) + 175))


def score_contribution_bar(row: pd.Series) -> go.Figure:
    """Stacked breakdown of one candidate's score into weighted points.

    The 6-slot palette carries a contrast warning on three slots, so the
    relief rule applies: segment values are direct-labelled where they fit, a
    legend is always present, and ``app.py`` renders the numeric table beside
    the chart.
    """
    fig = go.Figure()
    for key in COMPONENT_KEYS:
        points = float(row.get(f"contribution_{key}", 0.0))
        # Only label a segment when it is wide enough to hold the text.
        label = f"{points:.1f}" if points >= 6.0 else ""
        fig.add_trace(
            go.Bar(
                x=[points],
                y=["Illustrative score"],
                orientation="h",
                name=COMPONENT_LABELS[key],
                marker=dict(
                    color=COMPONENT_COLORS[key],
                    cornerradius=CORNER,
                    line=dict(color=PALETTE["surface"], width=SPACER),
                ),
                text=[label],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=12),
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{COMPONENT_LABELS[key]}</b><br>"
                    "Contributes %{x:.1f} of 100 points<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        title=f"Where {row['enzyme_name']}'s {row['overall_score']:.1f} points come from",
        bargap=0.6,
    )
    fig.update_xaxes(title="Points contributed (out of 100)", range=[0, 100])
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return _base_layout(fig, height=285)


def radar_chart(row: pd.Series, portfolio_mean: pd.Series) -> go.Figure:
    """Candidate profile against the portfolio average across the 6 components."""
    labels = [COMPONENT_LABELS[k] for k in COMPONENT_KEYS]
    closed = [*labels, labels[0]]

    def values(source) -> list[float]:
        vals = [float(source[k]) for k in COMPONENT_KEYS]
        return [*vals, vals[0]]

    fig = go.Figure()
    for name, source, color in (
        ("Portfolio average", portfolio_mean, PALETTE["series"][1]),
        (row["enzyme_name"], row, PALETTE["series"][0]),
    ):
        fig.add_trace(
            go.Scatterpolar(
                r=values(source),
                theta=closed,
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color,
                            line=dict(color=PALETTE["surface"], width=SPACER)),
                fill="toself",
                fillcolor=_rgba(color, 0.12),
                hovertemplate="<b>%{theta}</b><br>%{r:.2f} of 1.00<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        title="Component profile vs portfolio average",
        polar=dict(
            bgcolor=PALETTE["surface"],
            radialaxis=dict(
                range=[0, 1],
                gridcolor=PALETTE["grid"],
                linecolor=PALETTE["axis"],
                tickfont=dict(color=PALETTE["muted"], size=11),
            ),
            angularaxis=dict(
                gridcolor=PALETTE["grid"],
                linecolor=PALETTE["axis"],
                tickfont=dict(color=PALETTE["text_secondary"], size=11),
            ),
        ),
    )
    return _base_layout(fig, height=480)


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def component_comparison_bar(row: pd.Series, portfolio_mean: pd.Series) -> go.Figure:
    """Grouped bars: this candidate vs the portfolio average, per component."""
    labels = [COMPONENT_LABELS[k] for k in COMPONENT_KEYS]
    fig = go.Figure()
    for name, source, color in (
        (str(row["enzyme_name"]), row, PALETTE["series"][0]),
        ("Portfolio average", portfolio_mean, PALETTE["series"][1]),
    ):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[float(source[k]) for k in COMPONENT_KEYS],
                name=name,
                marker=dict(
                    color=color,
                    cornerradius=CORNER,
                    line=dict(color=PALETTE["surface"], width=SPACER),
                ),
                hovertemplate="<b>%{x}</b><br>%{y:.2f} of 1.00<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08,
        title="Component scores vs portfolio average",
    )
    fig.update_yaxes(title="Component score (0–1)", range=[0, 1.05])
    fig.update_xaxes(tickangle=-18)
    return _base_layout(fig, height=430)


# --------------------------------------------------------------------------
# Scenario analysis
# --------------------------------------------------------------------------
def rank_movement_chart(movement: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Dumbbell chart of rank changes between the base case and a scenario.

    Diverging encoding — blue for a candidate that moved up, red for one that
    moved down, neutral grey for no change. Direction is also carried by the
    arrow marker and by the direct label, so it never rests on colour alone.
    """
    data = movement.head(top_n).iloc[::-1]
    fig = go.Figure()
    legend_seen: set[str] = set()

    for _, r in data.iterrows():
        direction = r["direction"]
        color = {
            "Moved up": PALETTE["up"],
            "Moved down": PALETTE["down"],
            "Unchanged": PALETTE["flat"],
        }[direction]
        fig.add_trace(
            go.Scatter(
                x=[r["rank_base"], r["rank_scenario"]],
                y=[r["enzyme_name"], r["enzyme_name"]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["rank_base"]],
                y=[r["enzyme_name"]],
                mode="markers",
                name="Base-case rank",
                legendgroup="base",
                showlegend="base" not in legend_seen,
                marker=dict(
                    size=10, color=PALETTE["surface"], symbol="circle",
                    line=dict(color=color, width=2),
                ),
                hovertemplate="<b>%{y}</b><br>Base-case rank #%{x}<extra></extra>",
            )
        )
        legend_seen.add("base")
        symbol = {
            "Moved up": "triangle-left",
            "Moved down": "triangle-right",
            "Unchanged": "circle",
        }[direction]
        fig.add_trace(
            go.Scatter(
                x=[r["rank_scenario"]],
                y=[r["enzyme_name"]],
                mode="markers+text",
                name=direction,
                legendgroup=direction,
                showlegend=direction not in legend_seen,
                marker=dict(
                    size=13, color=color, symbol=symbol,
                    line=dict(color=PALETTE["surface"], width=SPACER),
                ),
                text=[
                    f"{r['rank_change']:+d}" if r["rank_change"] else "0"
                ],
                textposition="middle right",
                textfont=dict(color=PALETTE["text_secondary"], size=11),
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>Scenario rank #%{x}"
                    f"<br>Change {r['rank_change']:+d} places"
                    f"<br>Score change {r['score_change']:+.1f}<extra></extra>"
                ),
            )
        )
        legend_seen.add(direction)

    fig.update_layout(title=f"Rank movement — base case vs scenario (top {top_n})")
    fig.update_xaxes(
        title="Rank (1 = highest illustrative score)",
        autorange="reversed",
    )
    return _base_layout(fig, height=max(400, 26 * len(data) + 185))


def score_shift_chart(movement: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Score change per candidate under the scenario, as a diverging bar."""
    data = movement.head(top_n).iloc[::-1]
    colors = [
        PALETTE["up"] if v > 0 else (PALETTE["down"] if v < 0 else PALETTE["flat"])
        for v in data["score_change"]
    ]
    fig = go.Figure(
        go.Bar(
            x=data["score_change"],
            y=data["enzyme_name"],
            orientation="h",
            marker=dict(
                color=colors, cornerradius=CORNER,
                line=dict(color=PALETTE["surface"], width=SPACER),
            ),
            hovertemplate="<b>%{y}</b><br>Score change %{x:+.1f} points<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(title="Illustrative score change under the scenario")
    fig.update_xaxes(title="Change in illustrative score (points)")
    fig.add_vline(x=0, line_color=PALETTE["axis"], line_width=1)
    return _base_layout(fig, height=max(400, 26 * len(data) + 185), showlegend=False)


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------
def economics_bar(economics: dict) -> go.Figure:
    """Baseline vs prioritized screening cost. Both series are in EUR, one axis."""
    labels = ["Baseline: test the whole pool", "Prioritized: test the shortlist"]
    values = [economics["baseline_cost_eur"], economics["prioritized_cost_eur"]]
    counts = [economics["pool_size"], economics["n_selected"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=[PALETTE["muted"], PALETTE["series"][0]],
                cornerradius=CORNER,
                line=dict(color=PALETTE["surface"], width=SPACER),
            ),
            text=[f"EUR {v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(color=PALETTE["text_secondary"], size=12),
            customdata=counts,
            hovertemplate=(
                "<b>%{x}</b><br>%{customdata} candidates tested"
                "<br>Illustrative cost EUR %{y:,.0f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(title="Illustrative screening cost — baseline vs prioritized",
                      bargap=0.45)
    fig.update_yaxes(title="Illustrative cost (EUR)",
                     range=[0, max(values) * 1.18 if max(values) else 1])
    return _base_layout(fig, height=415, showlegend=False)


def sensitivity_chart(sensitivity: pd.DataFrame) -> go.Figure:
    """Cost curves across per-test cost assumptions. Crosshair hover enabled."""
    fig = go.Figure()
    series = [
        ("Baseline screening cost (EUR)", PALETTE["series"][1]),
        ("Prioritized screening cost (EUR)", PALETTE["series"][0]),
    ]
    for column, color in series:
        fig.add_trace(
            go.Scatter(
                x=sensitivity["Cost per test (EUR)"],
                y=sensitivity[column],
                mode="lines",
                name=column.replace(" (EUR)", ""),
                line=dict(color=color, width=2),
                hovertemplate="EUR %{y:,.0f}<extra>" + column.replace(" (EUR)", "") + "</extra>",
            )
        )
    # Direct-label the endpoint of each series rather than every point.
    for column, color in series:
        fig.add_trace(
            go.Scatter(
                x=[sensitivity["Cost per test (EUR)"].iloc[-1]],
                y=[sensitivity[column].iloc[-1]],
                mode="markers",
                marker=dict(size=8, color=color,
                            line=dict(color=PALETTE["surface"], width=SPACER)),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title="Sensitivity — how the illustrative cost depends on cost per test",
        hovermode="x unified",
    )
    fig.update_xaxes(title="Assumed cost per laboratory test (EUR)")
    fig.update_yaxes(title="Illustrative screening cost (EUR)")
    return _base_layout(fig, height=430)


# --------------------------------------------------------------------------
# Portfolio overview
# --------------------------------------------------------------------------
def condition_scatter(scored: pd.DataFrame, conditions: dict) -> go.Figure:
    """Where candidates sit against the requested temperature and pH.

    Three colour classes maximum (the first three palette slots validate on the
    all-pairs rule, which is what a scatter needs), plus a crosshair marker for
    the requested operating point.
    """
    fig = go.Figure()
    bands = [
        ("Prioritize for testing", PALETTE["series"][0], "circle"),
        ("Secondary queue", PALETTE["series"][1], "square"),
        ("Deprioritize for now", PALETTE["series"][2], "diamond"),
    ]
    for band, color, symbol in bands:
        subset = scored[scored["priority_band"] == band]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["estimated_temperature_optimum"],
                y=subset["estimated_ph_optimum"],
                mode="markers",
                name=band,
                marker=dict(
                    size=11, color=color, symbol=symbol,
                    line=dict(color=PALETTE["surface"], width=SPACER),
                ),
                customdata=np.stack(
                    [subset["enzyme_name"], subset["overall_score"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Temp optimum %{x:.1f} °C"
                    "<br>pH optimum %{y:.2f}"
                    "<br>Illustrative score %{customdata[1]:.1f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=conditions["temperature_c"], line_color=PALETTE["axis"], line_width=1)
    fig.add_hline(y=conditions["ph"], line_color=PALETTE["axis"], line_width=1)
    fig.add_trace(
        go.Scatter(
            x=[conditions["temperature_c"]],
            y=[conditions["ph"]],
            mode="markers+text",
            name="Your process conditions",
            marker=dict(size=15, color=PALETTE["text_primary"], symbol="x-thin",
                        line=dict(color=PALETTE["text_primary"], width=3)),
            text=["your process"],
            textposition="top center",
            textfont=dict(color=PALETTE["text_primary"], size=11),
            hovertemplate="Requested conditions<extra></extra>",
        )
    )
    fig.update_layout(title="Candidate optima against your requested conditions")
    fig.update_xaxes(title="Estimated temperature optimum (°C) — synthetic")
    fig.update_yaxes(title="Estimated pH optimum — synthetic")
    return _base_layout(fig, height=475)
