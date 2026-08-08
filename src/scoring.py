"""Transparent candidate-prioritization scoring for EnzySelect.

This module is a **decision-support heuristic**, not a biological model. It
takes user-declared process requirements and user-declared weights, and turns
them into a reproducible ranking. Every step is a plain arithmetic function
that can be read, argued with and changed.

Design rules kept deliberately strict:

* No component is allowed to be interpreted as a claim about real activity.
  ``structural_confidence`` is confidence in a *predicted shape*; it is not
  stability and it is not catalysis.
* Every component is normalized to 0-1 before weighting, so the weights mean
  what the sidebar says they mean.
* Two components (``performance`` and the cost half of ``feasibility``) are
  **pool-relative**: they are min-max normalized across the loaded candidate
  set. Change the pool and those scores change. This is documented in the UI
  because it is a real limitation, not an implementation detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COMPONENT_LABELS,
    DEFAULT_PROCESS_SUBWEIGHTS,
    DEFAULT_TOLERANCES,
    DEFAULT_WEIGHTS,
    PLDDT_FLOOR,
    PRIORITY_BANDS,
)

COMPONENT_KEYS = list(DEFAULT_WEIGHTS)

# Every input column that feeds a component score. A candidate missing any of
# these cannot be scored.
SCORED_INPUT_COLUMNS = [
    "estimated_temperature_optimum",
    "estimated_ph_optimum",
    "estimated_salinity_tolerance",
    "synthetic_degradation_rate",
    "catalytic_site_confidence",
    "plddt_mean",
    "estimated_production_cost",
    "estimated_expression_difficulty",
    "literature_evidence_level",
]


def unscoreable_mask(df: pd.DataFrame) -> np.ndarray:
    """True for rows that cannot be scored: missing, non-numeric or infinite.

    A candidate with a missing degradation rate must not be scored zero — that
    would be a claim about the candidate rather than a statement about the
    data. Such rows are excluded from the ranking and reported to the user
    instead.
    """
    columns = [c for c in SCORED_INPUT_COLUMNS if c in df.columns]
    if not columns:
        return np.ones(len(df), dtype=bool)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return ~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def gaussian_fit(values, target: float, tolerance: float):
    """Fit score in 0-1, peaking at ``target``.

    ``tolerance`` is the distance from the target at which the score falls to
    exp(-0.5) ~= 0.61. Symmetric: being 10 degrees too hot is penalized the
    same as 10 degrees too cold. That symmetry is an assumption, and a
    questionable one for real enzymes, which typically fall off much faster
    above their optimum than below it. It is documented in the README under
    limitations rather than silently corrected.
    """
    values = np.asarray(values, dtype=float)
    tolerance = max(float(tolerance), 1e-9)
    return np.exp(-0.5 * ((values - target) / tolerance) ** 2)


def threshold_fit(capability, requirement: float):
    """Fit for a 'must meet or exceed' variable such as salinity tolerance.

    Full credit once the candidate's tolerance meets the requirement; linear
    falloff below it. Excess tolerance earns no bonus, because tolerating more
    salt than the process contains is not an advantage.
    """
    capability = np.asarray(capability, dtype=float)
    if requirement <= 0:
        return np.ones_like(capability)
    return np.clip(capability / float(requirement), 0.0, 1.0)


def min_max_normalize(values):
    """Min-max to 0-1. A constant column maps to 0.5, not to NaN or 0."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def normalize_weights(weights: dict) -> dict:
    """Rescale weights to sum to 1 so the score stays on a 0-100 scale.

    If every weight is zero the weighting is meaningless, so fall back to a
    uniform weighting rather than dividing by zero.
    """
    clean = {k: max(float(weights.get(k, 0.0)), 0.0) for k in COMPONENT_KEYS}
    total = sum(clean.values())
    if total <= 0:
        return {k: 1.0 / len(COMPONENT_KEYS) for k in COMPONENT_KEYS}
    return {k: v / total for k, v in clean.items()}


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def compute_process_fit(
    df: pd.DataFrame,
    conditions: dict,
    tolerances: dict | None = None,
    subweights: dict | None = None,
) -> pd.DataFrame:
    """Temperature, pH and salinity fit, plus their weighted combination."""
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    subweights = {**DEFAULT_PROCESS_SUBWEIGHTS, **(subweights or {})}
    total = sum(subweights.values()) or 1.0

    out = pd.DataFrame(index=df.index)
    out["fit_temperature"] = gaussian_fit(
        df["estimated_temperature_optimum"],
        conditions["temperature_c"],
        tolerances["temperature_c"],
    )
    out["fit_ph"] = gaussian_fit(
        df["estimated_ph_optimum"], conditions["ph"], tolerances["ph"]
    )
    out["fit_salinity"] = threshold_fit(
        df["estimated_salinity_tolerance"], conditions["salinity_g_per_l"]
    )
    out["process_fit"] = (
        out["fit_temperature"] * subweights["temperature"]
        + out["fit_ph"] * subweights["ph"]
        + out["fit_salinity"] * subweights["salinity"]
    ) / total
    return out


def compute_components(
    df: pd.DataFrame,
    conditions: dict,
    tolerances: dict | None = None,
    subweights: dict | None = None,
) -> pd.DataFrame:
    """All six 0-1 component scores, plus the three process sub-fits."""
    out = compute_process_fit(df, conditions, tolerances, subweights)

    # Performance: log-scaled because the synthetic rates span two orders of
    # magnitude, then min-max normalized across the pool.
    rate = np.clip(df["synthetic_degradation_rate"].to_numpy(dtype=float), 1e-6, None)
    out["performance"] = min_max_normalize(np.log10(rate))

    # Catalytic-site confidence is already a 0-1 confidence value.
    out["catalytic_confidence"] = np.clip(
        df["catalytic_site_confidence"].to_numpy(dtype=float), 0.0, 1.0
    )

    # Structural confidence: pLDDT rescaled from the floor to 100.
    # This is confidence in a predicted fold. It is NOT stability or activity.
    out["structural_confidence"] = np.clip(
        (df["plddt_mean"].to_numpy(dtype=float) - PLDDT_FLOOR)
        / (100.0 - PLDDT_FLOOR),
        0.0,
        1.0,
    )

    # Feasibility: cheaper and easier to express scores higher. Cost is
    # log-scaled (it is generated log-normally) then pool-normalized;
    # difficulty is a fixed 1-5 ordinal so it needs no pool reference.
    cost = np.clip(df["estimated_production_cost"].to_numpy(dtype=float), 1e-6, None)
    cost_score = 1.0 - min_max_normalize(np.log10(cost))
    difficulty = df["estimated_expression_difficulty"].to_numpy(dtype=float)
    difficulty_score = 1.0 - np.clip((difficulty - 1.0) / 4.0, 0.0, 1.0)
    out["feasibility"] = 0.5 * cost_score + 0.5 * difficulty_score

    # Evidence: the synthetic 0-4 tier, rescaled.
    out["evidence"] = np.clip(
        df["literature_evidence_level"].to_numpy(dtype=float) / 4.0, 0.0, 1.0
    )

    return out


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def assign_priority_band(score: float) -> str:
    for threshold, label in PRIORITY_BANDS:
        if score >= threshold:
            return label
    return PRIORITY_BANDS[-1][1]


def score_candidates(
    df: pd.DataFrame,
    conditions: dict,
    weights: dict | None = None,
    tolerances: dict | None = None,
    subweights: dict | None = None,
) -> pd.DataFrame:
    """Return ``df`` with component scores, a 0-100 overall score and a rank.

    The returned frame is a copy; the input is never mutated.

    Candidates whose inputs are missing, non-numeric or infinite are **excluded
    from the ranking** rather than scored. Their ids are recorded in
    ``result.attrs["excluded_candidate_ids"]`` so the caller can tell the user
    which candidates were not assessed and why. Silently scoring them zero
    would present a data gap as a judgement about the enzyme.
    """
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)

    if df.empty:
        empty = df.copy()
        empty.attrs["excluded_candidate_ids"] = []
        empty.attrs["n_excluded"] = 0
        return empty

    unscoreable = unscoreable_mask(df)
    excluded_ids = (
        df.loc[unscoreable, "candidate_id"].astype(str).tolist()
        if "candidate_id" in df.columns
        else []
    )
    usable = df.loc[~unscoreable]

    if usable.empty:
        empty = usable.copy()
        empty.attrs["excluded_candidate_ids"] = excluded_ids
        empty.attrs["n_excluded"] = int(unscoreable.sum())
        return empty

    # Components are computed on the usable rows only, so the pool-relative
    # normalizations are not skewed by rows that will not be ranked.
    components = compute_components(usable, conditions, tolerances, subweights)

    scored = usable.copy()
    for column in components.columns:
        scored[column] = components[column].to_numpy()

    overall = np.zeros(len(scored), dtype=float)
    for key, weight in weights.items():
        contribution = components[key].to_numpy(dtype=float) * weight
        scored[f"contribution_{key}"] = contribution * 100.0
        overall += contribution

    scored["overall_score"] = np.round(overall * 100.0, 1)
    scored["rank"] = (
        scored["overall_score"].rank(ascending=False, method="first").astype(int)
    )
    scored["priority_band"] = scored["overall_score"].apply(assign_priority_band)

    result = scored.sort_values("rank").reset_index(drop=True)
    result.attrs["excluded_candidate_ids"] = excluded_ids
    result.attrs["n_excluded"] = int(unscoreable.sum())
    return result


def select_shortlist(
    scored: pd.DataFrame,
    max_candidates: int,
    budget_eur: float,
    cost_per_test_eur: float,
) -> pd.DataFrame:
    """Flag the candidates that fit the user's budget and headcount limits.

    Selection is 'rank order, take the top N', where N is bounded by three
    independent constraints. The binding one is reported so the user can see
    which limit is actually driving the shortlist.
    """
    carried = dict(scored.attrs)
    scored = scored.copy()
    scored.attrs.update(carried)
    n_pool = len(scored)
    cost_per_test_eur = max(float(cost_per_test_eur), 1e-9)

    affordable = int(np.floor(max(budget_eur, 0.0) / cost_per_test_eur))
    n_selected = int(max(0, min(int(max_candidates), affordable, n_pool)))

    scored["shortlisted"] = scored["rank"] <= n_selected
    scored["recommended_action"] = np.where(
        scored["shortlisted"],
        "Shortlist for laboratory testing",
        scored["priority_band"],
    )

    limits = {
        "maximum candidates to test": int(max_candidates),
        "testing budget": affordable,
        "candidates available": n_pool,
    }
    binding = min(limits, key=lambda name: limits[name])
    scored.attrs["n_selected"] = n_selected
    scored.attrs["binding_constraint"] = binding
    scored.attrs["affordable_tests"] = affordable
    return scored


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------
def contribution_table(row: pd.Series, weights: dict | None = None) -> pd.DataFrame:
    """Per-component breakdown of one candidate's 0-100 score."""
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)
    return pd.DataFrame(
        {
            "Component": [COMPONENT_LABELS[k] for k in COMPONENT_KEYS],
            "key": COMPONENT_KEYS,
            "Weight": [weights[k] for k in COMPONENT_KEYS],
            "Component score (0-1)": [float(row[k]) for k in COMPONENT_KEYS],
            "Points contributed": [
                float(row.get(f"contribution_{k}", row[k] * weights[k] * 100.0))
                for k in COMPONENT_KEYS
            ],
        }
    )


def explain_candidate(
    row: pd.Series, conditions: dict, weights: dict | None = None
) -> str:
    """Plain-language account of why this candidate scored what it scored."""
    table = contribution_table(row, weights).sort_values(
        "Points contributed", ascending=False
    )
    drivers = table.head(2)
    weakest = table[table["Weight"] > 0].tail(1)

    temp_delta = float(row["estimated_temperature_optimum"]) - conditions["temperature_c"]
    ph_delta = float(row["estimated_ph_optimum"]) - conditions["ph"]
    salt_ok = (
        float(row["estimated_salinity_tolerance"]) >= conditions["salinity_g_per_l"]
    )

    def direction(delta: float, unit: str) -> str:
        if abs(delta) < 0.05:
            return f"sits on your target {unit}"
        return f"is {abs(delta):.1f} {unit} {'above' if delta > 0 else 'below'} target"

    lines = [
        f"**{row['enzyme_name']}** scores **{row['overall_score']:.1f}/100** under "
        f"the current weights and process conditions, placing it at rank "
        f"**#{int(row['rank'])}**.",
        "",
        "**What drives the score**",
        "- The largest contributions come from "
        + " and ".join(
            f"*{r['Component'].lower()}* ({r['Points contributed']:.1f} points)"
            for _, r in drivers.iterrows()
        )
        + ".",
        f"- The weakest weighted area is *{weakest.iloc[0]['Component'].lower()}* "
        f"at {weakest.iloc[0]['Points contributed']:.1f} points.",
        "",
        "**Process fit against your conditions**",
        f"- Estimated temperature optimum {row['estimated_temperature_optimum']:.1f} °C "
        f"{direction(temp_delta, '°C')} — temperature fit "
        f"{row['fit_temperature']:.2f}.",
        f"- Estimated pH optimum {row['estimated_ph_optimum']:.2f} "
        f"{direction(ph_delta, 'pH units')} — pH fit {row['fit_ph']:.2f}.",
        f"- Estimated salinity tolerance {row['estimated_salinity_tolerance']:.0f} g/L "
        f"{'meets' if salt_ok else 'falls short of'} the "
        f"{conditions['salinity_g_per_l']:.0f} g/L you specified — salinity fit "
        f"{row['fit_salinity']:.2f}.",
        "",
        "**What this does not tell you**",
        "- The degradation rate behind the performance component is synthetic. "
        "It is not a measurement of this or any enzyme.",
        "- The structural component reflects confidence in a predicted fold "
        "only. It carries no information about activity, stability or yield.",
        "- This is a prioritization signal for ordering laboratory work. "
        "It requires experimental validation before it means anything.",
    ]
    return "\n".join(lines)


def rank_movement(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Join two rankings on candidate_id and report the rank change.

    Positive ``rank_change`` means the candidate moved *up* (toward rank 1).

    Both frames must describe the same candidates. A plain inner join would
    quietly drop anything present on only one side, and duplicate ids would
    fan out into a partial cross product — either way the scenario table would
    look plausible and be wrong, so both are rejected loudly instead.
    """
    for name, frame in (("base", before), ("scenario", after)):
        duplicated = frame["candidate_id"].duplicated().sum()
        if duplicated:
            raise ValueError(
                f"rank_movement: {duplicated} duplicate candidate_id value(s) "
                f"in the {name} ranking; ids must be unique."
            )

    only_before = set(before["candidate_id"]) - set(after["candidate_id"])
    only_after = set(after["candidate_id"]) - set(before["candidate_id"])
    if only_before or only_after:
        raise ValueError(
            "rank_movement: the two rankings describe different candidates "
            f"({len(only_before)} only in base, {len(only_after)} only in "
            "scenario). Compare rankings built from the same candidate set."
        )

    left = before[["candidate_id", "enzyme_name", "rank", "overall_score"]]
    right = after[["candidate_id", "rank", "overall_score"]]
    merged = left.merge(right, on="candidate_id", suffixes=("_base", "_scenario"))
    merged["rank_change"] = merged["rank_base"] - merged["rank_scenario"]
    merged["score_change"] = (
        merged["overall_score_scenario"] - merged["overall_score_base"]
    )
    merged["direction"] = np.select(
        [merged["rank_change"] > 0, merged["rank_change"] < 0],
        ["Moved up", "Moved down"],
        default="Unchanged",
    )
    return merged.sort_values("rank_base").reset_index(drop=True)
