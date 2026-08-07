"""EnzySelect — AI-assisted enzyme candidate prioritization (educational demo).

Run locally:

    streamlit run app.py

This is an educational prototype. It is not a validated scientific, medical,
environmental or industrial decision-making system, and the candidate
performance data it ranks is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CANDIDATES_CSV,
    COMPONENT_LABELS,
    DEFAULT_CONDITIONS,
    DEFAULT_ECONOMICS,
    DEFAULT_PROCESS_SUBWEIGHTS,
    DEFAULT_TOLERANCES,
    DEFAULT_WEIGHTS,
    HEADLINE_DISCLAIMER,
    PALETTE,
    REFERENCES_CSV,
    SCIENTIFIC_INTEGRITY_NOTES,
    SCORE_CAVEAT,
    STRUCTURE_LINK_CAVEAT,
    SYNTHETIC_DATA_NOTICE,
)
from src.economics import (
    ECONOMICS_CAVEATS,
    assumptions_frame,
    compute_economics,
    format_eur,
    sensitivity_table,
)
from src.scoring import (
    COMPONENT_KEYS,
    contribution_table,
    explain_candidate,
    normalize_weights,
    rank_movement,
    score_candidates,
    select_shortlist,
)
from src.structures import (
    get_structure,
    placeholder_backbone_pdb,
    structure_summary,
    viewer_html,
)
import src.visualizations as viz

st.set_page_config(
    page_title="EnzySelect — enzyme candidate prioritization (demo)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
      .es-note {{
        border-left: 3px solid {PALETTE['axis']};
        padding: 0.35rem 0 0.35rem 0.85rem;
        color: {PALETTE['text_secondary']};
        font-size: 0.87rem;
        margin: 0.4rem 0 0.9rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def note(text: str) -> None:
    """A recessive caveat line — present, readable, not shouting."""
    st.markdown(f'<div class="es-note">{text}</div>', unsafe_allow_html=True)


def render_raw_html(html: str, height: int = 440) -> None:
    """Embed self-contained HTML+JS (the py3Dmol viewer).

    ``st.html`` is the current API; ``st.components.v1.html`` is deprecated but
    kept as a fallback so the app still works on older Streamlit versions.
    """
    try:
        st.html(html, unsafe_allow_javascript=True)
    except TypeError:  # Streamlit older than the unsafe_allow_javascript flag
        st.components.v1.html(html, height=height)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "candidate_id", "enzyme_name", "structure_family", "organism",
    "sequence_length", "plddt_mean", "catalytic_site_confidence",
    "estimated_temperature_optimum", "estimated_ph_optimum",
    "estimated_salinity_tolerance", "synthetic_degradation_rate",
    "estimated_expression_difficulty", "estimated_production_cost",
    "literature_evidence_level", "experimental_status",
]


@st.cache_data(show_spinner=False)
def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES_CSV)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"candidates.csv is missing columns: {missing}")
    return df


@st.cache_data(show_spinner=False)
def load_references() -> pd.DataFrame:
    if REFERENCES_CSV.is_file():
        return pd.read_csv(REFERENCES_CSV)
    return pd.DataFrame()


@st.cache_data(show_spinner="Resolving structure…", ttl=3600)
def cached_structure(uniprot_id: str, allow_network: bool) -> dict:
    """Cache the plain-dict form of a StructureResult (hashable for Streamlit)."""
    result = get_structure(uniprot_id, allow_network=allow_network)
    return {
        "available": result.available,
        "source": result.source,
        "message": result.message,
        "pdb_text": result.pdb_text,
        "entry_url": result.entry_url,
        "model_url": result.model_url,
        "real_mean_plddt": result.real_mean_plddt,
        "is_real_structure": result.is_real_structure,
    }


# --------------------------------------------------------------------------
# Sidebar — section 2: process configuration
# --------------------------------------------------------------------------
def sidebar_controls(n_candidates: int) -> dict:
    st.sidebar.title("Process configuration")
    st.sidebar.caption(
        "Target material: **PET plastic** (fixed in this demonstration)."
    )

    st.sidebar.subheader("Operating conditions")
    temperature = st.sidebar.slider(
        "Operating temperature (°C)", 20.0, 90.0,
        float(DEFAULT_CONDITIONS["temperature_c"]), 0.5,
        help="Candidates score higher when their estimated temperature "
             "optimum is close to this value.",
    )
    ph = st.sidebar.slider(
        "Operating pH", 4.0, 11.0, float(DEFAULT_CONDITIONS["ph"]), 0.1,
        help="Candidates score higher when their estimated pH optimum is "
             "close to this value.",
    )
    salinity = st.sidebar.slider(
        "Process salinity (g/L NaCl)", 0.0, 150.0,
        float(DEFAULT_CONDITIONS["salinity_g_per_l"]), 1.0,
        help="Candidates score full marks once their estimated salinity "
             "tolerance meets this requirement.",
    )

    st.sidebar.subheader("Testing constraints")
    cost_per_test = st.sidebar.number_input(
        "Cost per laboratory test (EUR)", 100.0, 100_000.0,
        float(DEFAULT_ECONOMICS["cost_per_test_eur"]), 100.0,
    )
    budget = st.sidebar.number_input(
        "Maximum laboratory testing budget (EUR)", 0.0, 5_000_000.0,
        float(DEFAULT_ECONOMICS["budget_eur"]), 1000.0,
    )
    max_candidates = st.sidebar.slider(
        "Maximum number of candidates to test", 1, min(30, n_candidates),
        int(DEFAULT_ECONOMICS["max_candidates"]),
    )

    st.sidebar.subheader("Scoring weights")
    st.sidebar.caption(
        "Weights are renormalized to sum to 1, so only their relative size "
        "matters. Changing them changes the ranking."
    )
    weights = {}
    for key, default in DEFAULT_WEIGHTS.items():
        weights[key] = st.sidebar.slider(
            COMPONENT_LABELS[key], 0.0, 1.0, float(default), 0.05
        )
    if st.sidebar.button("Reset weights to default", width="stretch"):
        st.session_state.clear()
        st.rerun()

    with st.sidebar.expander("Advanced — fit tolerances"):
        st.caption(
            "Tolerance is the distance from your target at which the fit "
            "score falls to about 0.61."
        )
        temp_tol = st.slider("Temperature tolerance (± °C)", 2.0, 25.0,
                             float(DEFAULT_TOLERANCES["temperature_c"]), 0.5)
        ph_tol = st.slider("pH tolerance (± pH units)", 0.2, 3.0,
                           float(DEFAULT_TOLERANCES["ph"]), 0.1)
        st.caption("Relative importance inside the process-fit component:")
        sub_t = st.slider("… temperature", 0.0, 1.0,
                          DEFAULT_PROCESS_SUBWEIGHTS["temperature"], 0.05)
        sub_p = st.slider("… pH", 0.0, 1.0, DEFAULT_PROCESS_SUBWEIGHTS["ph"], 0.05)
        sub_s = st.slider("… salinity", 0.0, 1.0,
                          DEFAULT_PROCESS_SUBWEIGHTS["salinity"], 0.05)

    with st.sidebar.expander("Advanced — laboratory throughput"):
        weeks_per_cycle = st.slider("Weeks per laboratory cycle", 0.5, 12.0,
                                    float(DEFAULT_ECONOMICS["weeks_per_cycle"]), 0.5)
        tests_per_cycle = st.slider("Tests run in parallel per cycle", 1, 24,
                                    int(DEFAULT_ECONOMICS["tests_per_cycle"]))

    allow_network = st.sidebar.toggle(
        "Allow external API calls", value=True,
        help="When off, the app uses only locally cached structures. The "
             "ranking never depends on network access.",
    )

    return {
        "conditions": {
            "temperature_c": temperature, "ph": ph, "salinity_g_per_l": salinity,
        },
        "weights": weights,
        "tolerances": {"temperature_c": temp_tol, "ph": ph_tol},
        "subweights": {"temperature": sub_t, "ph": sub_p, "salinity": sub_s},
        "cost_per_test": cost_per_test,
        "budget": budget,
        "max_candidates": max_candidates,
        "weeks_per_cycle": weeks_per_cycle,
        "tests_per_cycle": tests_per_cycle,
        "allow_network": allow_network,
    }


# --------------------------------------------------------------------------
# Section 1 — executive summary
# --------------------------------------------------------------------------
def render_executive_summary(scored: pd.DataFrame, economics: dict) -> None:
    st.subheader("Executive summary")
    top = scored.iloc[0]
    n_selected = scored.attrs.get("n_selected", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candidates assessed", f"{len(scored)}", border=True)
    c2.metric("Recommended for testing", f"{n_selected}", border=True)
    c3.metric(
        "Top candidate", str(top["enzyme_name"]),
        delta=f"{top['overall_score']:.1f}/100 illustrative", delta_color="off",
        border=True,
    )
    c4.metric(
        "Average illustrative score",
        f"{scored['overall_score'].mean():.1f}/100", border=True,
    )
    c5.metric(
        "Potential avoided testing cost",
        format_eur(economics["cost_avoided_eur"]),
        delta=f"{economics['pct_reduction']:.0f}% fewer tests", delta_color="off",
        border=True,
    )

    note(
        f"The shortlist is bound by <b>{scored.attrs.get('binding_constraint','')}</b>. "
        "Avoided cost is illustrative arithmetic on your assumptions — "
        "<b>not a validated ROI estimate</b>, and not a measured saving."
    )
    st.warning(HEADLINE_DISCLAIMER)


# --------------------------------------------------------------------------
# Section 3 — ranked candidates
# --------------------------------------------------------------------------
BAND_TINT = {
    "Prioritize for testing": "background-color: #cde2fb;",
    "Secondary queue": "background-color: #e9f1fc;",
    "Deprioritize for now": "background-color: #fcfcfb;",
}

RANK_COLUMNS = {
    "rank": "Rank",
    "enzyme_name": "Candidate",
    "structure_family": "Structural family",
    "overall_score": "Illustrative score",
    "process_fit": "Process fit",
    "synthetic_degradation_rate": "Synthetic degradation rate",
    "structural_confidence": "Structural confidence",
    "feasibility": "Production feasibility",
    "recommended_action": "Recommended action",
}


def ranked_display_frame(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[list(RANK_COLUMNS)].rename(columns=RANK_COLUMNS)


def render_ranked_candidates(scored: pd.DataFrame, conditions: dict) -> None:
    st.subheader("Ranked candidates")
    note(
        "Priority tint encodes the score band and is always accompanied by "
        "the written action, so no information is carried by colour alone. "
        + SCORE_CAVEAT
    )

    display = ranked_display_frame(scored)
    styler = (
        display.style.format(
            {
                "Illustrative score": "{:.1f}",
                "Process fit": "{:.2f}",
                "Synthetic degradation rate": "{:.2f}",
                "Structural confidence": "{:.2f}",
                "Production feasibility": "{:.2f}",
            }
        )
        .apply(
            lambda row: [BAND_TINT.get(
                scored.loc[row.name, "priority_band"], ""
            )] * len(row),
            axis=1,
        )
    )
    st.dataframe(styler, hide_index=True, height=430)

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(viz.top_candidates_bar(scored), theme=None)
    with right:
        st.plotly_chart(viz.condition_scatter(scored, conditions), theme=None)

    st.caption(
        "Synthetic degradation rate is expressed in arbitrary demonstration "
        "units (mg PET per mg enzyme per hour). It is generated, not measured."
    )


# --------------------------------------------------------------------------
# Section 4 — candidate deep dive
# --------------------------------------------------------------------------
def render_structure_panel(row: pd.Series, allow_network: bool) -> None:
    accession = str(row.get("structure_reference_uniprot_id", "") or "")
    st.markdown("**Predicted structure — family reference**")
    st.info(STRUCTURE_LINK_CAVEAT)

    if not accession:
        st.write("No structural family reference is recorded for this candidate.")
        return

    st.markdown(
        f"Family **{row['structure_family']}** is anchored to "
        f"**{row.get('structure_reference_name', accession)}** "
        f"(*{row.get('structure_reference_organism', 'organism unknown')}*), "
        f"UniProt `{accession}`."
    )

    result = cached_structure(accession, allow_network)
    links = [f"[AlphaFold DB entry]({result['entry_url']})",
             f"[UniProt entry](https://www.uniprot.org/uniprotkb/{accession}/entry)"]
    if result["model_url"]:
        links.append(f"[Model coordinates]({result['model_url']})")
    st.markdown(" · ".join(links))

    if result["available"] and result["pdb_text"]:
        summary = structure_summary(result["pdb_text"])
        a, b, c = st.columns(3)
        a.metric("Residues in model", f"{summary['residues_resolved']}")
        if summary["mean_plddt_from_file"] is not None:
            b.metric("Mean pLDDT (real, reference)",
                     f"{summary['mean_plddt_from_file']:.1f}")
        if summary["pct_residues_plddt_70_plus"] is not None:
            c.metric("Residues at pLDDT ≥ 70",
                     f"{summary['pct_residues_plddt_70_plus']:.0f}%")
        st.caption(f"Source: {result['message']}")

        # The viewer is opt-in on purpose. Streamlit renders every tab on each
        # page load, and the py3Dmol payload is ~190 kB of markup plus a WebGL
        # context. Building that unconditionally makes the whole page heavy —
        # and can lock up the browser renderer — for a panel most users never
        # scroll to. Gating it behind a toggle keeps the app responsive.
        show_viewer = st.toggle(
            "Load the interactive 3D viewer",
            value=False,
            key=f"viewer_{row['candidate_id']}",
            help="Loads ~190 kB of viewer code and 3Dmol.js from a CDN. The "
                 "metadata above and the links are available without it.",
        )
        if show_viewer:
            html = viewer_html(result["pdb_text"])
            if html:
                render_raw_html(html, height=440)
                st.caption(
                    "Cartoon coloured by per-residue pLDDT (red = low "
                    "confidence, blue = high). This is confidence in predicted "
                    "geometry only — it is not activity, stability or "
                    "industrial suitability. The viewer loads 3Dmol.js from a "
                    "CDN, so it needs internet even when the coordinates came "
                    "from the local cache."
                )
            else:
                st.caption(
                    "The 3D viewer could not be initialised; the links above "
                    "still work."
                )
    else:
        st.warning(result["message"])
        if st.checkbox("Show a labelled geometric placeholder instead",
                       key=f"placeholder_{row['candidate_id']}"):
            html = viewer_html(placeholder_backbone_pdb(60), style="trace")
            if html:
                render_raw_html(html, height=380)
            st.error(
                "**This is not a protein.** It is an idealised alpha-helix "
                "drawn from textbook geometry, shown only to demonstrate that "
                "the viewer works offline. It represents no candidate and no "
                "real molecule."
            )


def render_deep_dive(scored: pd.DataFrame, settings: dict) -> None:
    st.subheader("Candidate deep dive")
    labels = [
        f"#{int(r['rank'])} — {r['enzyme_name']} ({r['overall_score']:.1f}/100)"
        for _, r in scored.iterrows()
    ]
    choice = st.selectbox("Select a candidate", labels, index=0)
    row = scored.iloc[labels.index(choice)]

    meta_left, meta_right = st.columns(2)
    with meta_left:
        st.markdown("**Candidate metadata**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": [
                        "Candidate ID", "Structural family", "Simulated provenance",
                        "Sequence length", "UniProt accession", "Experimental status",
                        "Evidence level (synthetic)", "Data provenance",
                    ],
                    "Value": [
                        row["candidate_id"], row["structure_family"], row["organism"],
                        f"{int(row['sequence_length'])} aa",
                        "none — this candidate is synthetic",
                        row["experimental_status"],
                        row.get("literature_evidence_label", row["literature_evidence_level"]),
                        row["data_provenance"],
                    ],
                }
            ),
            hide_index=True,
        )
    with meta_right:
        st.markdown("**Synthetic estimates**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Estimate (synthetic)": [
                        "Temperature optimum", "pH optimum", "Salinity tolerance",
                        "Degradation rate", "pLDDT (candidate, synthetic)",
                        "Catalytic-site confidence", "Expression difficulty",
                        "Production cost",
                    ],
                    "Value": [
                        f"{row['estimated_temperature_optimum']:.1f} °C",
                        f"{row['estimated_ph_optimum']:.2f}",
                        f"{row['estimated_salinity_tolerance']:.0f} g/L",
                        f"{row['synthetic_degradation_rate']:.2f} (demo units)",
                        f"{row['plddt_mean']:.1f}",
                        f"{row['catalytic_site_confidence']:.2f}",
                        f"{int(row['estimated_expression_difficulty'])} of 5",
                        f"EUR {row['estimated_production_cost']:,.0f} / g",
                    ],
                }
            ),
            hide_index=True,
        )
    note(
        "Every value in the right-hand table was generated by a random "
        "process. None of it is a measurement of this or any enzyme."
    )

    st.markdown("---")
    st.markdown("**Score breakdown**")
    st.plotly_chart(viz.score_contribution_bar(row), theme=None)
    breakdown = contribution_table(row, settings["weights"])
    st.dataframe(
        breakdown.drop(columns=["key"]).style.format(
            {"Weight": "{:.0%}", "Component score (0-1)": "{:.3f}",
             "Points contributed": "{:.1f}"}
        ),
        hide_index=True,
    )

    portfolio_mean = scored[COMPONENT_KEYS].mean()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.radar_chart(row, portfolio_mean), theme=None)
    with right:
        st.plotly_chart(viz.component_comparison_bar(row, portfolio_mean), theme=None)

    st.markdown("---")
    st.markdown("**Why this candidate scored what it scored**")
    st.markdown(explain_candidate(row, settings["conditions"], settings["weights"]))

    st.markdown("---")
    render_structure_panel(row, settings["allow_network"])

    with st.expander("Evidence and limitations for this candidate", expanded=False):
        for line in SCIENTIFIC_INTEGRITY_NOTES:
            st.markdown(f"- {line}")


# --------------------------------------------------------------------------
# Section 5 — scenario analysis
# --------------------------------------------------------------------------
def render_scenario(df: pd.DataFrame, base: pd.DataFrame, settings: dict) -> None:
    st.subheader("Scenario analysis")
    st.caption(
        "Change the process assumptions and see how the ranking responds. A "
        "ranking that moves a lot under a small change is telling you the "
        "shortlist is fragile, which is useful information about the method "
        "rather than about the enzymes."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    base_conditions = settings["conditions"]
    scenario_conditions = {
        "temperature_c": c1.slider("Scenario temperature (°C)", 20.0, 90.0,
                                   float(base_conditions["temperature_c"]), 0.5,
                                   key="sc_temp"),
        "ph": c2.slider("Scenario pH", 4.0, 11.0, float(base_conditions["ph"]),
                        0.1, key="sc_ph"),
        "salinity_g_per_l": c3.slider("Scenario salinity (g/L)", 0.0, 150.0,
                                      float(base_conditions["salinity_g_per_l"]),
                                      1.0, key="sc_salt"),
    }
    scenario_cost = c4.number_input("Scenario cost per test (EUR)", 100.0, 100_000.0,
                                    float(settings["cost_per_test"]), 100.0,
                                    key="sc_cost")
    scenario_max = c5.slider("Scenario max candidates", 1, min(30, len(df)),
                             int(settings["max_candidates"]), key="sc_max")

    scenario = score_candidates(df, scenario_conditions, settings["weights"],
                                settings["tolerances"], settings["subweights"])
    scenario = select_shortlist(scenario, scenario_max, settings["budget"],
                                scenario_cost)
    movement = rank_movement(base, scenario)

    base_top = set(base[base["shortlisted"]]["candidate_id"])
    scen_top = set(scenario[scenario["shortlisted"]]["candidate_id"])
    retained = len(base_top & scen_top)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shortlist size — scenario", f"{scenario.attrs['n_selected']}",
              border=True)
    m2.metric("Shortlist overlap with base case",
              f"{retained} of {max(len(base_top), 1)}", border=True)
    m3.metric("Largest upward move",
              f"{int(movement['rank_change'].max()):+d} places", border=True)
    m4.metric("Largest downward move",
              f"{int(movement['rank_change'].min()):+d} places", border=True)

    if len(base_top) and retained < len(base_top):
        st.warning(
            f"{len(base_top) - retained} of {len(base_top)} shortlisted "
            "candidates change under this scenario. The shortlist is sensitive "
            "to the process assumptions you entered — treat it as one option "
            "among several, not as a settled answer."
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.rank_movement_chart(movement), theme=None)
    with right:
        st.plotly_chart(viz.score_shift_chart(movement), theme=None)

    st.markdown("**Rank movement table**")
    table = movement[[
        "enzyme_name", "rank_base", "rank_scenario", "rank_change",
        "overall_score_base", "overall_score_scenario", "score_change", "direction",
    ]].rename(columns={
        "enzyme_name": "Candidate", "rank_base": "Base rank",
        "rank_scenario": "Scenario rank", "rank_change": "Change (places)",
        "overall_score_base": "Base score", "overall_score_scenario": "Scenario score",
        "score_change": "Score change", "direction": "Direction",
    })
    st.dataframe(
        table.style.format({
            "Base score": "{:.1f}", "Scenario score": "{:.1f}",
            "Score change": "{:+.1f}", "Change (places)": "{:+d}",
        }),
        hide_index=True, height=320,
    )
    return scenario_conditions


# --------------------------------------------------------------------------
# Section 6 — economics
# --------------------------------------------------------------------------
def render_economics(scored: pd.DataFrame, settings: dict, economics: dict) -> None:
    st.subheader("Illustrative economics")
    st.info(
        "**Not a validated ROI estimate.** These figures are arithmetic on "
        "assumptions you entered, applied to synthetic candidate data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline screening cost",
              format_eur(economics["baseline_cost_eur"]),
              help=f"{economics['pool_size']} candidates × "
                   f"EUR {economics['cost_per_test_eur']:,.0f}", border=True)
    c2.metric("Prioritized screening cost",
              format_eur(economics["prioritized_cost_eur"]),
              help=f"{economics['n_selected']} candidates tested", border=True)
    c3.metric("Potential avoided testing cost",
              format_eur(economics["cost_avoided_eur"]),
              delta=f"{economics['pct_reduction']:.1f}% illustrative reduction",
              delta_color="off", border=True)
    c4.metric("Estimated time saved",
              f"{economics['weeks_saved']:.1f} weeks",
              delta=f"{economics['baseline_weeks']:.0f} → "
                    f"{economics['prioritized_weeks']:.0f} weeks",
              delta_color="off", border=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.economics_bar(economics), theme=None)
    with right:
        lo = max(economics["cost_per_test_eur"] * 0.25, 100.0)
        hi = economics["cost_per_test_eur"] * 2.5
        sensitivity = sensitivity_table(
            economics["pool_size"], economics["n_selected"],
            np.linspace(lo, hi, 12), settings["weeks_per_cycle"],
            settings["tests_per_cycle"],
        )
        st.plotly_chart(viz.sensitivity_chart(sensitivity), theme=None)

    st.markdown("**Sensitivity table**")
    st.dataframe(
        sensitivity.style.format("{:,.0f}"), hide_index=True, height=260
    )

    st.markdown("**How to read these numbers**")
    for caveat in ECONOMICS_CAVEATS:
        st.markdown(f"- {caveat}")


# --------------------------------------------------------------------------
# Section 7 — downloads
# --------------------------------------------------------------------------
def render_downloads(scored: pd.DataFrame, settings: dict, economics: dict) -> None:
    st.subheader("Download")
    st.caption(
        "Every export carries a provenance column or an assumptions row, so a "
        "shortlist cannot be read later without the caveats that produced it."
    )

    shortlist = scored[scored["shortlisted"]].copy()
    shortlist["export_note"] = (
        "Educational prototype. Synthetic data. Requires experimental "
        "validation. Not a validated ROI or scientific result."
    )

    breakdown = scored[
        ["rank", "candidate_id", "enzyme_name", "overall_score"]
        + COMPONENT_KEYS
        + [f"contribution_{k}" for k in COMPONENT_KEYS]
        + ["fit_temperature", "fit_ph", "fit_salinity", "priority_band"]
    ].copy()

    assumptions = assumptions_frame(
        economics, settings["conditions"], normalize_weights(settings["weights"])
    )
    business_case = pd.DataFrame([economics])

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Ranked shortlist (CSV)", shortlist.to_csv(index=False).encode("utf-8"),
        "enzyselect_shortlist.csv", "text/csv", width="stretch",
    )
    c2.download_button(
        "Score breakdown (CSV)", breakdown.to_csv(index=False).encode("utf-8"),
        "enzyselect_score_breakdown.csv", "text/csv", width="stretch",
    )
    c3.download_button(
        "Scenario assumptions (CSV)", assumptions.to_csv(index=False).encode("utf-8"),
        "enzyselect_assumptions.csv", "text/csv", width="stretch",
    )
    c4.download_button(
        "Business-case calculations (CSV)",
        business_case.to_csv(index=False).encode("utf-8"),
        "enzyselect_business_case.csv", "text/csv", width="stretch",
    )

    st.markdown("**Assumptions that will be exported**")
    st.dataframe(assumptions, hide_index=True, height=380)


# --------------------------------------------------------------------------
# Method & limitations
# --------------------------------------------------------------------------
def render_method(references: pd.DataFrame, settings: dict) -> None:
    st.subheader("Method, data and limitations")

    st.markdown("**Scientific integrity notes**")
    for line in SCIENTIFIC_INTEGRITY_NOTES:
        st.markdown(f"- {line}")

    st.markdown("**Scoring formula**")
    weights = normalize_weights(settings["weights"])
    st.latex(
        r"\text{score} = 100 \times \sum_{i} w_i \cdot s_i,"
        r"\qquad \sum_i w_i = 1,\; s_i \in [0,1]"
    )
    st.dataframe(
        pd.DataFrame({
            "Component": [COMPONENT_LABELS[k] for k in COMPONENT_KEYS],
            "Weight in use": [weights[k] for k in COMPONENT_KEYS],
            "What it is": [
                "Gaussian fit on temperature and pH, threshold fit on salinity.",
                "Synthetic degradation rate, log-scaled and pool-normalized.",
                "Synthetic confidence that a catalytic site is correctly placed.",
                "Synthetic pLDDT rescaled from 50 to 100. Geometry confidence only.",
                "Lower production cost and expression difficulty score higher.",
                "Synthetic evidence tier, 0 to 4.",
            ],
        }).style.format({"Weight in use": "{:.0%}"}),
        hide_index=True,
    )
    st.caption(SCORE_CAVEAT)

    st.markdown("**Known limitations of this method**")
    st.markdown(
        "- Temperature and pH fit are **symmetric**. Real enzymes typically "
        "lose activity far faster above their optimum than below it, so a "
        "symmetric penalty flatters candidates that would be operating too hot.\n"
        "- Performance and the cost half of feasibility are **pool-relative** "
        "(min-max normalized). Adding or removing candidates changes the "
        "scores of the others.\n"
        "- The six components are treated as **independent and additive**. In "
        "reality they trade off and interact; a linear weighted sum cannot "
        "express 'high activity is worthless if the protein will not express'.\n"
        "- The weights are a **stated preference**, not a fitted model. There "
        "is no ground truth in this dataset to fit them against.\n"
        "- The dataset is synthetic, so **no accuracy claim of any kind** can "
        "be made about the ranking."
    )

    st.markdown("**Data provenance**")
    st.info(SYNTHETIC_DATA_NOTICE)
    if not references.empty:
        st.markdown(
            "The structural family anchors below are **real** public database "
            "records, retrieved from the UniProt and AlphaFold REST APIs. They "
            "carry no synthetic performance values."
        )
        st.dataframe(
            references[[
                "structure_family", "reference_uniprot_id", "reference_protein_name",
                "reference_organism", "reference_sequence_length",
                "reference_plddt_mean_real", "metadata_source", "verified_on",
            ]].rename(columns={
                "structure_family": "Family",
                "reference_uniprot_id": "UniProt",
                "reference_protein_name": "Protein name (real)",
                "reference_organism": "Organism (real)",
                "reference_sequence_length": "Length",
                "reference_plddt_mean_real": "Mean pLDDT (real)",
                "metadata_source": "Retrieved from",
                "verified_on": "Verified on",
            }),
            hide_index=True,
        )
        st.caption(
            "The real mean pLDDT column describes AlphaFold's confidence in "
            "its prediction for these characterized proteins. It is not "
            "evidence of PET-degrading activity, and it is unrelated to the "
            "synthetic pLDDT values attached to the demo candidates."
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("EnzySelect")
    st.caption(
        "AI-assisted enzyme candidate prioritization for PET plastic "
        "degradation — educational prototype, demonstration only."
    )

    try:
        df = load_candidates()
    except FileNotFoundError:
        st.error(
            "`data/candidates.csv` was not found. Generate the synthetic "
            "dataset first:\n\n```bash\npython data/generate_data.py\n```"
        )
        st.stop()
    except ValueError as exc:
        st.error(f"The candidate dataset failed validation: {exc}")
        st.stop()

    references = load_references()
    settings = sidebar_controls(len(df))

    scored = score_candidates(df, settings["conditions"], settings["weights"],
                              settings["tolerances"], settings["subweights"])
    scored = select_shortlist(scored, settings["max_candidates"],
                              settings["budget"], settings["cost_per_test"])

    economics = compute_economics(
        pool_size=len(df),
        n_selected=scored.attrs["n_selected"],
        cost_per_test_eur=settings["cost_per_test"],
        weeks_per_cycle=settings["weeks_per_cycle"],
        tests_per_cycle=settings["tests_per_cycle"],
    )

    render_executive_summary(scored, economics)
    st.markdown("---")

    tabs = st.tabs([
        "Ranked candidates", "Candidate deep dive", "Scenario analysis",
        "Economics", "Download", "Method & limitations",
    ])
    with tabs[0]:
        render_ranked_candidates(scored, settings["conditions"])
    with tabs[1]:
        render_deep_dive(scored, settings)
    with tabs[2]:
        render_scenario(df, scored, settings)
    with tabs[3]:
        render_economics(scored, settings, economics)
    with tabs[4]:
        render_downloads(scored, settings, economics)
    with tabs[5]:
        render_method(references, settings)


if __name__ == "__main__":
    main()
