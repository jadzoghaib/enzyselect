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

from src.config import DEFAULT_CONDITIONS, DEFAULT_WEIGHTS
from src.scoring import (
    COMPONENT_KEYS,
    contribution_table,
    explain_candidate,
    gaussian_fit,
    min_max_normalize,
    normalize_weights,
    rank_movement,
    score_candidates,
    select_shortlist,
    threshold_fit,
    unscoreable_mask,
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


def test_rank_movement_rejects_mismatched_candidate_sets(candidates):
    """An inner join would silently drop the missing candidates instead."""
    full = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    partial = score_candidates(candidates.head(20), DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    with pytest.raises(ValueError, match="different candidates"):
        rank_movement(full, partial)


def test_rank_movement_rejects_duplicate_ids(candidates):
    """Duplicates would fan out into a partial cross product."""
    doubled = pd.concat([candidates, candidates.head(1)], ignore_index=True)
    scored = score_candidates(doubled, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        rank_movement(scored, scored)


# --------------------------------------------------------------------------
# Unscoreable candidates
#
# A missing input is a gap in the data, not a fact about the enzyme. These
# rows must be excluded and reported, never scored zero and never crash.
# --------------------------------------------------------------------------
def test_missing_value_does_not_crash_the_ranking(candidates):
    broken = candidates.copy()
    broken.loc[0, "synthetic_degradation_rate"] = np.nan
    scored = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert len(scored) == len(candidates) - 1
    assert scored["overall_score"].notna().all()


def test_excluded_candidates_are_reported_by_id(candidates):
    broken = candidates.copy()
    broken.loc[0, "plddt_mean"] = np.nan
    broken.loc[3, "estimated_production_cost"] = np.inf
    scored = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert scored.attrs["n_excluded"] == 2
    assert set(scored.attrs["excluded_candidate_ids"]) == {
        candidates.loc[0, "candidate_id"],
        candidates.loc[3, "candidate_id"],
    }


def test_an_excluded_candidate_is_not_scored_zero(candidates):
    broken = candidates.copy()
    broken.loc[0, "catalytic_site_confidence"] = np.nan
    scored = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert candidates.loc[0, "candidate_id"] not in set(scored["candidate_id"])


def test_non_numeric_junk_is_treated_as_unscoreable(candidates):
    broken = candidates.copy()
    broken["estimated_ph_optimum"] = broken["estimated_ph_optimum"].astype(object)
    broken.loc[1, "estimated_ph_optimum"] = "not a number"
    scored = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert scored.attrs["n_excluded"] == 1


def test_all_rows_unscoreable_returns_empty_not_an_exception(candidates):
    broken = candidates.copy()
    broken["plddt_mean"] = np.nan
    scored = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert scored.empty
    assert scored.attrs["n_excluded"] == len(candidates)


def test_exclusion_does_not_skew_pool_relative_normalization(candidates):
    """A dropped row must not shift the min-max range of the survivors."""
    clean = score_candidates(candidates.iloc[1:], DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    broken = candidates.copy()
    broken.loc[0, "synthetic_degradation_rate"] = np.nan
    with_gap = score_candidates(broken, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    pd.testing.assert_series_equal(
        clean.set_index("candidate_id")["overall_score"].sort_index(),
        with_gap.set_index("candidate_id")["overall_score"].sort_index(),
    )


def test_unscoreable_mask_flags_only_the_bad_rows(candidates):
    broken = candidates.copy()
    broken.loc[7, "estimated_temperature_optimum"] = np.nan
    mask = unscoreable_mask(broken)
    assert mask.sum() == 1 and bool(mask[7])


def test_clean_dataset_excludes_nobody(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert scored.attrs["n_excluded"] == 0
    assert len(scored) == len(candidates)


# --------------------------------------------------------------------------
# Explanation text
# --------------------------------------------------------------------------
def test_explanation_names_the_candidate_and_its_score(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    row = scored.iloc[0]
    text = explain_candidate(row, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    assert str(row["enzyme_name"]) in text
    assert f"{row['overall_score']:.1f}" in text


def test_explanation_always_states_the_limitations(candidates):
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS)
    for i in (0, 5, len(scored) - 1):
        text = explain_candidate(scored.iloc[i], DEFAULT_CONDITIONS, DEFAULT_WEIGHTS).lower()
        assert "synthetic" in text
        assert "requires experimental validation" in text
        assert "not activity" in text or "no information about activity" in text


def test_explanation_survives_all_zero_weights(candidates):
    weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    scored = score_candidates(candidates, DEFAULT_CONDITIONS, weights)
    assert explain_candidate(scored.iloc[0], DEFAULT_CONDITIONS, weights)
