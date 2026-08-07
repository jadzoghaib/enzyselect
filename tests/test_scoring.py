"""Tests for the scoring engine.

The point of these is not coverage for its own sake. Each one pins a property
that, if it broke silently, would make the ranking wrong in a way nobody would
notice by looking at the dashboard.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CONDITIONS, DEFAULT_WEIGHTS  # noqa: E402
from src.scoring import (  # noqa: E402
    COMPONENT_KEYS,
    contribution_table,
    gaussian_fit,
    min_max_normalize,
    normalize_weights,
    rank_movement,
    score_candidates,
    select_shortlist,
    threshold_fit,
)


@pytest.fixture(scope="module")
def candidates() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "candidates.csv")


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def test_gaussian_fit_peaks_at_the_target():
    assert gaussian_fit([50.0], 50.0, 8.0)[0] == pytest.approx(1.0)


def test_gaussian_fit_hits_exp_half_at_one_tolerance():
    assert gaussian_fit([58.0], 50.0, 8.0)[0] == pytest.approx(math.exp(-0.5))


def test_gaussian_fit_is_symmetric_about_the_target():
    below, above = gaussian_fit([42.0, 58.0], 50.0, 8.0)
    assert below == pytest.approx(above)


def test_gaussian_fit_survives_zero_tolerance():
    # Must not divide by zero; an exact match still scores 1.
    assert gaussian_fit([50.0, 51.0], 50.0, 0.0).tolist() == [1.0, 0.0]


def test_threshold_fit_gives_full_credit_when_requirement_is_met():
    assert threshold_fit([50.0, 100.0], 50.0).tolist() == [1.0, 1.0]


def test_threshold_fit_does_not_reward_excess_tolerance():
    # Tolerating 3x the salt in the process is not 3x as good.
    assert threshold_fit([150.0], 50.0)[0] == 1.0


def test_threshold_fit_ramps_down_below_the_requirement():
    assert threshold_fit([25.0], 50.0)[0] == pytest.approx(0.5)


def test_threshold_fit_with_no_requirement_is_neutral():
    assert threshold_fit([0.0, 80.0], 0.0).tolist() == [1.0, 1.0]


def test_min_max_normalize_maps_a_constant_column_to_one_half():
    # Not NaN and not 0 — a constant feature must not silently zero a component.
    assert min_max_normalize([7.0, 7.0, 7.0]).tolist() == [0.5, 0.5, 0.5]


def test_min_max_normalize_spans_zero_to_one():
    out = min_max_normalize([2.0, 4.0, 6.0])
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_weights_sums_to_one():
    weights = normalize_weights({k: 2.0 for k in DEFAULT_WEIGHTS})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_normalize_weights_falls_back_to_uniform_when_all_zero():
    weights = normalize_weights({k: 0.0 for k in DEFAULT_WEIGHTS})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(set(np.round(list(weights.values()), 9))) == 1


def test_normalize_weights_ignores_negative_input():
    weights = normalize_weights({**DEFAULT_WEIGHTS, "feasibility": -5.0})
    assert weights["feasibility"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def test_scores_stay_on_the_zero_to_hundred_scale(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert scored["overall_score"].between(0, 100).all()


def test_ranks_are_a_permutation_of_one_to_n(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert sorted(scored["rank"]) == list(range(1, len(candidates) + 1))


def test_scoring_does_not_mutate_the_input_frame(candidates):
    before = candidates.copy(deep=True)
    score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    pd.testing.assert_frame_equal(candidates, before)


def test_contributions_sum_to_the_overall_score(candidates):
    """The headline invariant: the breakdown must explain the whole score."""
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    total = scored[[f"contribution_{k}" for k in COMPONENT_KEYS]].sum(axis=1)
    assert np.allclose(total, scored["overall_score"], atol=0.05)


def test_contribution_table_matches_the_row(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    table = contribution_table(scored.iloc[0], DEFAULT_WEIGHTS)
    assert table["Points contributed"].sum() == pytest.approx(
        scored.iloc[0]["overall_score"], abs=0.05
    )
    assert table["Weight"].sum() == pytest.approx(1.0)


def test_ranking_responds_to_the_operating_temperature(candidates):
    hot = score_candidates(
        candidates, {**DEFAULT_CONDITIONS, "temperature_c": 80.0}, DEFAULT_WEIGHTS
    )
    cold = score_candidates(
        candidates, {**DEFAULT_CONDITIONS, "temperature_c": 30.0}, DEFAULT_WEIGHTS
    )
    assert hot.iloc[0]["candidate_id"] != cold.iloc[0]["candidate_id"]


def test_a_pure_process_fit_weighting_favours_the_best_fitting_candidate(candidates):
    weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    weights["process_fit"] = 1.0
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, weights)
    assert scored.iloc[0]["process_fit"] == pytest.approx(scored["process_fit"].max())


def test_empty_input_returns_empty_output():
    empty = pd.DataFrame(columns=["candidate_id"])
    assert score_candidates(empty, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS).empty


# --------------------------------------------------------------------------
# Shortlist selection
# --------------------------------------------------------------------------
def test_shortlist_respects_the_candidate_cap(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    result = select_shortlist(scored, max_candidates=5, budget_eur=10**9,
                              cost_per_test_eur=1000.0)
    assert result["shortlisted"].sum() == 5
    assert result.attrs["binding_constraint"] == "maximum candidates to test"


def test_shortlist_respects_the_budget(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    # Budget affords exactly 3 tests, which is tighter than the cap of 20.
    result = select_shortlist(scored, max_candidates=20, budget_eur=3000.0,
                              cost_per_test_eur=1000.0)
    assert result["shortlisted"].sum() == 3
    assert result.attrs["binding_constraint"] == "testing budget"


def test_shortlist_takes_the_highest_ranked_candidates(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    result = select_shortlist(scored, 6, 10**9, 1000.0)
    assert set(result[result["shortlisted"]]["rank"]) == set(range(1, 7))


def test_zero_budget_shortlists_nobody(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    result = select_shortlist(scored, 10, 0.0, 1000.0)
    assert result["shortlisted"].sum() == 0


def test_shortlist_cannot_exceed_the_pool(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    result = select_shortlist(scored, 10_000, 10**9, 1.0)
    assert result["shortlisted"].sum() == len(candidates)


# --------------------------------------------------------------------------
# Scenario comparison
# --------------------------------------------------------------------------
def test_rank_movement_sign_convention(candidates):
    """Positive rank_change must mean 'moved up', i.e. toward rank 1."""
    base = score_candidates(candidates, {**DEFAULT_CONDITIONS, "temperature_c": 75.0},
                            DEFAULT_WEIGHTS)
    scenario = score_candidates(candidates, {**DEFAULT_CONDITIONS, "temperature_c": 35.0},
                                DEFAULT_WEIGHTS)
    movement = rank_movement(base, scenario)
    improved = movement[movement["rank_change"] > 0]
    assert (improved["rank_scenario"] < improved["rank_base"]).all()
    assert (movement[movement["direction"] == "Unchanged"]["rank_change"] == 0).all()


def test_rank_movement_against_itself_is_all_zero(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    movement = rank_movement(scored, scored)
    assert (movement["rank_change"] == 0).all()
    assert (movement["score_change"] == 0).all()
