"""Tests for the Plotly figures.

Two of these encode layout bugs that only showed up when the app was opened in
a browser: axis labels truncated to a single character, and the chart title
colliding with the legend. Both are invisible to a test that merely checks the
figure builds, so they are pinned explicitly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.visualizations as viz
from src.config import COMPONENT_COLORS, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS
from src.economics import compute_economics, sensitivity_table
from src.scoring import (
    COMPONENT_KEYS,
    rank_movement,
    score_candidates,
    select_shortlist,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "candidates.csv")
    ranked = score_candidates(df, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    return select_shortlist(ranked, 8, 60_000.0, 4_200.0)


@pytest.fixture(scope="module")
def movement(scored) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "candidates.csv")
    scenario = score_candidates(
        df, {**DEFAULT_CONDITIONS, "temperature_c": 35.0}, DEFAULT_WEIGHTS
    )
    return rank_movement(scored, scenario)


@pytest.fixture(scope="module")
def figures(scored, movement) -> dict[str, go.Figure]:
    row = scored.iloc[0]
    mean = scored[COMPONENT_KEYS].mean()
    econ = compute_economics(50, 8, 4_200.0, 3.0, 8)
    sens = sensitivity_table(50, 8, [1000.0, 2000.0, 3000.0], 3.0, 8)
    return {
        "top_candidates_bar": viz.top_candidates_bar(scored),
        "score_contribution_bar": viz.score_contribution_bar(row),
        "radar_chart": viz.radar_chart(row, mean),
        "component_comparison_bar": viz.component_comparison_bar(row, mean),
        "rank_movement_chart": viz.rank_movement_chart(movement),
        "score_shift_chart": viz.score_shift_chart(movement),
        "economics_bar": viz.economics_bar(econ),
        "sensitivity_chart": viz.sensitivity_chart(sens),
        "condition_scatter": viz.condition_scatter(scored, DEFAULT_CONDITIONS),
    }


def test_every_figure_builds(figures):
    for name, fig in figures.items():
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) > 0, name


def test_no_figure_uses_a_second_y_axis(figures):
    """Dual-axis charts invent correlations. There must be none anywhere."""
    for name, fig in figures.items():
        assert "yaxis2" not in fig.layout, name
        assert "xaxis2" not in fig.layout, name


def test_top_margin_leaves_room_for_title_and_legend(figures):
    """Regression: at t=48 the title overlapped the horizontal legend."""
    for name, fig in figures.items():
        assert fig.layout.margin.t >= 90, name


def test_cartesian_axes_use_automargin(figures):
    """Regression: without automargin, candidate names truncated to one char."""
    for name, fig in figures.items():
        if fig.layout.xaxis.title.text or fig.layout.yaxis.title.text:
            assert fig.layout.xaxis.automargin is True, name
            assert fig.layout.yaxis.automargin is True, name


def test_gridlines_are_solid_not_dashed(figures):
    for name, fig in figures.items():
        if fig.layout.xaxis.griddash is not None:
            assert fig.layout.xaxis.griddash == "solid", name


def test_component_colours_are_fixed_and_distinct():
    assert set(COMPONENT_COLORS) == set(COMPONENT_KEYS)
    assert len(set(COMPONENT_COLORS.values())) == len(COMPONENT_KEYS)


def test_colour_follows_the_component_not_the_ranking(scored):
    """A component keeps its hue whichever candidate is being shown."""
    first = viz.score_contribution_bar(scored.iloc[0])
    other = viz.score_contribution_bar(scored.iloc[10])
    a = {t.name: t.marker.color for t in first.data}
    b = {t.name: t.marker.color for t in other.data}
    assert a == b


def test_multi_series_figures_show_a_legend(figures):
    for name in ("radar_chart", "component_comparison_bar", "sensitivity_chart"):
        assert figures[name].layout.showlegend is True, name


def test_stacked_contributions_span_the_full_score(scored):
    row = scored.iloc[0]
    fig = viz.score_contribution_bar(row)
    total = sum(float(trace.x[0]) for trace in fig.data)
    assert total == pytest.approx(float(row["overall_score"]), abs=0.05)


def test_top_candidates_bar_separates_shortlisted_from_the_rest(scored):
    fig = viz.top_candidates_bar(scored, top_n=12)
    names = {t.name for t in fig.data}
    assert names == {"Shortlisted for testing", "Below the shortlist cut"}


def test_figures_survive_a_single_candidate(scored):
    """Degenerate pool: min-max normalization collapses; nothing may raise."""
    one = scored.head(1)
    assert isinstance(viz.top_candidates_bar(one), go.Figure)
    assert isinstance(
        viz.condition_scatter(one, DEFAULT_CONDITIONS), go.Figure
    )
