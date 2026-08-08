"""Tests for the illustrative economics calculations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.economics import (
    assumptions_frame,
    compute_economics,
    format_eur,
    sensitivity_table,
)


def test_baseline_and_prioritized_cost_arithmetic():
    result = compute_economics(50, 8, 4200.0, 3.0, 8)
    assert result["baseline_cost_eur"] == 50 * 4200
    assert result["prioritized_cost_eur"] == 8 * 4200
    assert result["cost_avoided_eur"] == (50 - 8) * 4200


def test_percentage_reduction():
    result = compute_economics(50, 10, 1000.0, 1.0, 5)
    assert result["pct_reduction"] == pytest.approx(80.0)
    assert result["tests_reduced"] == 40


def test_cycles_round_up_to_whole_laboratory_runs():
    """9 tests at a capacity of 8 costs two full cycles, not 1.125."""
    result = compute_economics(9, 9, 100.0, 2.0, 8)
    assert result["baseline_cycles"] == 2
    assert result["baseline_weeks"] == 4.0


def test_no_selection_means_no_cycles_and_no_time():
    result = compute_economics(50, 0, 1000.0, 3.0, 8)
    assert result["prioritized_cycles"] == 0
    assert result["prioritized_weeks"] == 0.0
    assert result["cost_avoided_eur"] == 50_000


def test_selection_is_clipped_to_the_pool():
    result = compute_economics(10, 999, 1000.0, 1.0, 4)
    assert result["n_selected"] == 10
    assert result["cost_avoided_eur"] == 0


def test_empty_pool_does_not_divide_by_zero():
    result = compute_economics(0, 0, 1000.0, 1.0, 4)
    assert result["pct_reduction"] == 0.0
    assert result["baseline_weeks"] == 0.0


def test_zero_parallel_capacity_is_treated_as_one():
    result = compute_economics(4, 4, 100.0, 1.0, 0)
    assert result["tests_per_cycle"] == 1
    assert result["baseline_cycles"] == 4


def test_avoided_cost_is_linear_in_cost_per_test():
    """The headline saving is a restatement of the user's cost assumption."""
    table = sensitivity_table(50, 8, [1000.0, 2000.0], 1.0, 8)
    avoided = table["Potential avoided testing cost (EUR)"].tolist()
    assert avoided[1] == pytest.approx(avoided[0] * 2)


def test_sensitivity_table_shape():
    table = sensitivity_table(50, 8, range(1000, 5001, 1000), 1.0, 8)
    assert len(table) == 5
    assert "Potential avoided testing cost (EUR)" in table.columns


def test_assumptions_frame_is_arrow_safe_and_records_provenance():
    economics = compute_economics(50, 8, 4200.0, 3.0, 8)
    frame = assumptions_frame(
        economics, {"temperature_c": 65.0, "ph": 8.0, "salinity_g_per_l": 20.0},
        {"process_fit": 0.3, "performance": 0.7},
    )
    # Mixed-type object columns break Arrow serialization in Streamlit.
    assert frame["Value"].map(type).eq(str).all()
    assert (frame["Assumption"] == "Data provenance").any()
    assert frame.loc[frame["Assumption"] == "Data provenance", "Value"].iloc[0] == "synthetic"


def test_assumptions_frame_flags_the_roi_caveat():
    economics = compute_economics(50, 8, 4200.0, 3.0, 8)
    frame = assumptions_frame(economics, {"temperature_c": 1.0, "ph": 7.0,
                                          "salinity_g_per_l": 0.0}, {})
    notes = " ".join(frame["Unit / note"])
    assert "not a validated ROI estimate" in notes


@pytest.mark.parametrize(
    "value,expected",
    [(950.0, "EUR 950"), (12_500.0, "EUR 12.5k"), (2_400_000.0, "EUR 2.40M")],
)
def test_currency_formatting(value, expected):
    assert format_eur(value) == expected
