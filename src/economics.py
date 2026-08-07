"""Illustrative screening-economics calculations for EnzySelect.

These are arithmetic consequences of user-entered assumptions. They are **not
a validated ROI estimate** and they are not a forecast. The numbers describe a
demonstration scenario in which a laboratory tests fewer candidates because a
heuristic ranked them, and nothing more.

The one thing worth being honest about, which this module encodes explicitly
in :data:`ECONOMICS_CAVEATS`: "cost avoided" only exists if the shortlist
still contains the candidate you were looking for. A ranking that saves 80% of
the testing budget and drops the only viable enzyme has not saved anything --
it has bought a cheaper failure. The savings figure and the risk of a missed
candidate are the same decision viewed twice, so the UI shows them together.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ECONOMICS_CAVEATS = [
    "Potential avoided testing cost is arithmetic on your assumptions, not a "
    "measured saving and not a validated ROI estimate.",
    "The figure assumes the full screening pool would otherwise have been "
    "tested exhaustively. If a laboratory would have prioritized informally "
    "anyway, the true baseline is lower and the avoided cost is overstated.",
    "It also assumes the shortlist retains the candidates worth finding. "
    "Cost avoided and the risk of screening out a viable candidate are two "
    "views of the same decision — a cheaper screen that misses the hit has "
    "saved nothing.",
    "Cost per test, cycle length and throughput are user-entered planning "
    "assumptions; no laboratory quotation or timesheet informs them.",
    "Because the underlying candidate data is synthetic, no result here "
    "supports a real procurement, production or investment decision.",
]


def compute_economics(
    pool_size: int,
    n_selected: int,
    cost_per_test_eur: float,
    weeks_per_cycle: float,
    tests_per_cycle: int,
) -> dict:
    """Baseline vs prioritized screening cost and elapsed time.

    Time is modelled as whole laboratory cycles: a laboratory that can run
    ``tests_per_cycle`` assays in parallel needs ``ceil(n / tests_per_cycle)``
    cycles, each taking ``weeks_per_cycle`` weeks. Rounding up matters here --
    testing 9 candidates with a capacity of 8 costs two full cycles, not 1.125.
    """
    pool_size = max(int(pool_size), 0)
    n_selected = int(np.clip(n_selected, 0, pool_size))
    cost_per_test_eur = max(float(cost_per_test_eur), 0.0)
    weeks_per_cycle = max(float(weeks_per_cycle), 0.0)
    tests_per_cycle = max(int(tests_per_cycle), 1)

    baseline_cost = pool_size * cost_per_test_eur
    prioritized_cost = n_selected * cost_per_test_eur
    cost_avoided = baseline_cost - prioritized_cost

    tests_reduced = pool_size - n_selected
    pct_reduction = (tests_reduced / pool_size * 100.0) if pool_size else 0.0

    baseline_cycles = math.ceil(pool_size / tests_per_cycle) if pool_size else 0
    prioritized_cycles = math.ceil(n_selected / tests_per_cycle) if n_selected else 0
    baseline_weeks = baseline_cycles * weeks_per_cycle
    prioritized_weeks = prioritized_cycles * weeks_per_cycle

    return {
        "pool_size": pool_size,
        "n_selected": n_selected,
        "cost_per_test_eur": cost_per_test_eur,
        "baseline_cost_eur": baseline_cost,
        "prioritized_cost_eur": prioritized_cost,
        "cost_avoided_eur": cost_avoided,
        "tests_reduced": tests_reduced,
        "pct_reduction": pct_reduction,
        "baseline_cycles": baseline_cycles,
        "prioritized_cycles": prioritized_cycles,
        "baseline_weeks": baseline_weeks,
        "prioritized_weeks": prioritized_weeks,
        "weeks_saved": baseline_weeks - prioritized_weeks,
        "tests_per_cycle": tests_per_cycle,
        "weeks_per_cycle": weeks_per_cycle,
    }


def sensitivity_table(
    pool_size: int,
    n_selected: int,
    cost_range_eur,
    weeks_per_cycle: float,
    tests_per_cycle: int,
) -> pd.DataFrame:
    """Recompute the economics across a range of per-test costs.

    Cost avoided is linear in cost per test, which is exactly the point: the
    headline number is a restatement of the user's own cost assumption, so the
    chart should make that dependency visible rather than hide it behind one
    impressive figure.
    """
    rows = []
    for cost in cost_range_eur:
        result = compute_economics(
            pool_size, n_selected, cost, weeks_per_cycle, tests_per_cycle
        )
        rows.append(
            {
                "Cost per test (EUR)": result["cost_per_test_eur"],
                "Baseline screening cost (EUR)": result["baseline_cost_eur"],
                "Prioritized screening cost (EUR)": result["prioritized_cost_eur"],
                "Potential avoided testing cost (EUR)": result["cost_avoided_eur"],
            }
        )
    return pd.DataFrame(rows)


def assumptions_frame(economics: dict, conditions: dict, weights: dict) -> pd.DataFrame:
    """Flat, downloadable record of everything the user chose.

    Exported alongside the ranking so a shortlist can never be read without
    the assumptions that produced it.
    """
    rows = [
        ("Target material", "PET plastic", "fixed in this demo"),
        ("Operating temperature", f"{conditions['temperature_c']:.1f}", "degrees C"),
        ("Operating pH", f"{conditions['ph']:.2f}", "pH units"),
        ("Salinity requirement", f"{conditions['salinity_g_per_l']:.1f}", "g/L NaCl"),
        ("Screening pool size", economics["pool_size"], "candidates"),
        ("Candidates shortlisted", economics["n_selected"], "candidates"),
        ("Cost per laboratory test", economics["cost_per_test_eur"], "EUR"),
        ("Tests per laboratory cycle", economics["tests_per_cycle"], "tests"),
        ("Weeks per laboratory cycle", economics["weeks_per_cycle"], "weeks"),
        ("Baseline screening cost", economics["baseline_cost_eur"], "EUR, illustrative"),
        (
            "Prioritized screening cost",
            economics["prioritized_cost_eur"],
            "EUR, illustrative",
        ),
        (
            "Potential avoided testing cost",
            economics["cost_avoided_eur"],
            "EUR, illustrative — not a validated ROI estimate",
        ),
        (
            "Illustrative screening-cost reduction",
            f"{economics['pct_reduction']:.1f}",
            "%, illustrative",
        ),
        ("Estimated time saved", economics["weeks_saved"], "weeks, illustrative"),
    ]
    rows += [
        (f"Weight — {key}", f"{value:.3f}", "scoring weight, normalized")
        for key, value in weights.items()
    ]
    rows.append(
        (
            "Data provenance",
            "synthetic",
            "candidate performance values are generated, not measured",
        )
    )
    frame = pd.DataFrame(rows, columns=["Assumption", "Value", "Unit / note"])
    # The Value column deliberately mixes text and numbers, which Arrow cannot
    # serialize as a mixed object column. Cast to string so the table renders
    # and the CSV export stays human-readable.
    frame["Value"] = frame["Value"].astype(str)
    return frame


def format_eur(value: float) -> str:
    """Compact currency formatting for metric tiles."""
    if abs(value) >= 1_000_000:
        return f"EUR {value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"EUR {value / 1_000:,.1f}k"
    return f"EUR {value:,.0f}"
