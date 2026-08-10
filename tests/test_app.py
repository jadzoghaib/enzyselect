"""End-to-end tests for the Streamlit application.

These drive the real app through Streamlit's ``AppTest`` harness. They are the
only tests that exercise the wiring between the scoring engine, the economics
and the interface — which is where every defect found in the quality pass
actually lived, none of them in the arithmetic.

They are slower than the rest of the suite (each ``run()`` executes the whole
script) so the read-only assertions share one module-scoped run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TIMEOUT = 300


def fresh_app() -> AppTest:
    """A new AppTest with Streamlit's caches cleared.

    ``load_candidates`` is memoized with ``st.cache_data``, so a test that
    points the app at a different CSV would otherwise be served the previous
    test's dataframe.
    """
    import streamlit as st

    st.cache_data.clear()
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = fresh_app()
    at.run()
    return at


def widget_by_label(collection, fragment: str):
    return next(w for w in collection if fragment.lower() in w.label.lower())


# --------------------------------------------------------------------------
# The app renders at all
# --------------------------------------------------------------------------
def test_app_runs_without_exceptions(app):
    assert not app.exception, [e.message for e in app.exception]


def test_all_seven_sections_are_present(app):
    assert len(app.tabs) == 6  # plus the executive summary above the tabs
    titles = " ".join(m.label for m in app.metric)
    assert "Candidates assessed" in titles
    assert "Top candidate" in titles


def test_headline_disclaimer_is_shown(app):
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "Educational prototype" in warnings
    assert "not a validated scientific" in warnings


def test_executive_summary_agrees_with_the_scoring_engine(app):
    """Cross-check the displayed numbers against an independent computation."""
    from src.config import CANDIDATES_CSV, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS
    from src.scoring import score_candidates, select_shortlist

    df = pd.read_csv(CANDIDATES_CSV)
    expected = select_shortlist(
        score_candidates(df, DEFAULT_CONDITIONS, DEFAULT_WEIGHTS),
        8, 60_000.0, 4_200.0,
    )
    assert app.metric[0].value == str(len(df))
    assert app.metric[2].value == str(expected.iloc[0]["enzyme_name"])


def test_download_buttons_are_offered(app):
    labels = [b.label for b in app.download_button]
    assert any("shortlist" in x.lower() for x in labels)
    assert any("breakdown" in x.lower() for x in labels)
    assert any("assumption" in x.lower() for x in labels)
    assert any("business" in x.lower() for x in labels)


def test_method_tab_publishes_the_integrity_notes(app):
    from src.config import SCIENTIFIC_INTEGRITY_NOTES

    body = " ".join(str(m.value) for m in app.markdown)
    assert SCIENTIFIC_INTEGRITY_NOTES[0][:45] in body


# --------------------------------------------------------------------------
# The controls actually drive the model
# --------------------------------------------------------------------------
def test_operating_temperature_changes_the_top_candidate():
    at = fresh_app()
    at.run()
    hot = at.metric[2].value
    widget_by_label(at.slider, "Operating temperature").set_value(35.0).run()
    assert at.metric[2].value != hot
    assert not at.exception


def test_pool_size_drives_the_avoided_cost():
    at = fresh_app()
    at.run()
    small = at.metric[4].value
    widget_by_label(at.number_input, "Original screening pool").set_value(500).run()
    assert at.metric[4].value != small
    assert not at.exception


def test_zero_weights_fall_back_to_uniform_without_crashing():
    at = fresh_app()
    at.run()
    for slider in at.slider:
        if slider.label in {
            "Process-condition fit", "Synthetic degradation performance",
            "Catalytic-site confidence", "Structural confidence (pLDDT)",
            "Production feasibility", "Evidence level (synthetic)",
        }:
            slider.set_value(0.0)
    at.run()
    assert not at.exception
    assert at.metric[2].value


def test_reset_weights_leaves_other_widgets_alone():
    """Regression: the button used to call st.session_state.clear()."""
    at = fresh_app()
    at.run()
    widget_by_label(at.number_input, "Original screening pool").set_value(321).run()
    widget_by_label(at.button, "Reset weights").click().run()
    assert not at.exception
    assert widget_by_label(at.number_input, "Original screening pool").value == 321


def test_viewer_is_opt_in():
    """Regression, measured twice: building it eagerly renders a 0x0 canvas.

    Streamlit builds every tab on page load, so an eagerly-created viewer is
    measured while its panel is hidden and stays blank. It must stay off by
    default, and the panel must advertise that it exists.
    """
    at = fresh_app()
    at.run()
    assert widget_by_label(at.toggle, "3D viewer").value is False
    captions = " ".join(str(c.value) for c in at.caption)
    assert "3D structure" in captions and "switch it on" in captions.lower()
    assert not at.exception


def test_viewer_renders_when_switched_on():
    at = fresh_app()
    at.run()
    widget_by_label(at.toggle, "3D viewer").set_value(True).run()
    assert not at.exception
    # The metadata and links must be present either way.
    body = " ".join(str(m.value) for m in at.markdown)
    assert "AlphaFold DB entry" in body


def test_viewer_markup_is_memoized_not_rebuilt(monkeypatch):
    """Regression: rebuilding ~190 kB of WebGL markup on every rerun froze the
    browser. Now that the viewer renders by default, a second render of the
    same structure must be served from cache rather than rebuilt.

    Identity is not the test — ``st.cache_data`` hands back a copy — so this
    counts calls to the underlying builder instead.
    """
    import app as enzyselect_app
    from src.structures import get_structure

    result = get_structure("Q6A0I4", allow_network=False)
    if not result.pdb_text:
        pytest.skip("no cached structure available")

    enzyselect_app.cached_viewer_html.clear()
    builds: list[str] = []
    original = enzyselect_app.viewer_html

    def counting(pdb_text, style="cartoon"):
        builds.append(style)
        return original(pdb_text, style=style)

    monkeypatch.setattr(enzyselect_app, "viewer_html", counting)
    first = enzyselect_app.cached_viewer_html(result.pdb_text, "Q6A0I4")
    second = enzyselect_app.cached_viewer_html(result.pdb_text, "Q6A0I4")

    assert first == second
    assert len(builds) == 1, "viewer markup was rebuilt instead of cached"
    assert "3dmolviewer" in first


def test_scenario_controls_recompute_without_error():
    at = fresh_app()
    at.run()
    widget_by_label(at.slider, "Scenario temperature").set_value(30.0).run()
    widget_by_label(at.slider, "Scenario pH").set_value(6.0).run()
    assert not at.exception


def test_selecting_another_candidate_renders_its_deep_dive():
    at = fresh_app()
    at.run()
    box = at.selectbox[0]
    box.select(box.options[3]).run()
    assert not at.exception
    body = " ".join(str(m.value) for m in at.markdown)
    assert "requires experimental validation" in body.lower()


def test_disabling_network_keeps_the_app_working():
    at = fresh_app()
    at.run()
    widget_by_label(at.toggle, "external API").set_value(False).run()
    assert not at.exception
    assert at.metric[0].value


# --------------------------------------------------------------------------
# Data-quality paths
# --------------------------------------------------------------------------
def _app_pointed_at(tmp_path: Path, frame: pd.DataFrame, monkeypatch) -> AppTest:
    csv = tmp_path / "candidates.csv"
    frame.to_csv(csv, index=False)
    monkeypatch.setattr("src.config.CANDIDATES_CSV", csv)
    return fresh_app()


def test_unscoreable_candidates_are_reported_not_crashed(tmp_path, monkeypatch):
    from src.config import CANDIDATES_CSV

    df = pd.read_csv(CANDIDATES_CSV)
    df.loc[0, "synthetic_degradation_rate"] = None
    df.loc[5, "plddt_mean"] = None
    at = _app_pointed_at(tmp_path, df, monkeypatch)
    at.run()
    assert not at.exception
    assert at.metric[0].value == str(len(df) - 2)
    assert any("could not be scored" in str(w.value) for w in at.warning)


def test_duplicate_candidate_ids_are_refused(tmp_path, monkeypatch):
    from src.config import CANDIDATES_CSV

    df = pd.read_csv(CANDIDATES_CSV)
    at = _app_pointed_at(tmp_path, pd.concat([df, df.head(1)]), monkeypatch)
    at.run()
    assert any("duplicate candidate_id" in str(e.value) for e in at.error)


def test_missing_dataset_gives_actionable_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.CANDIDATES_CSV", tmp_path / "absent.csv")
    at = fresh_app()
    at.run()
    errors = " ".join(str(e.value) for e in at.error)
    assert "generate_data.py" in errors


def test_a_missing_required_column_is_named_in_the_error(tmp_path, monkeypatch):
    from src.config import CANDIDATES_CSV

    df = pd.read_csv(CANDIDATES_CSV).drop(columns=["plddt_mean"])
    at = _app_pointed_at(tmp_path, df, monkeypatch)
    at.run()
    errors = " ".join(str(e.value) for e in at.error)
    assert "missing columns" in errors and "plddt_mean" in errors


def test_a_candidate_without_a_family_reference_says_so(tmp_path, monkeypatch):
    from src.config import CANDIDATES_CSV

    df = pd.read_csv(CANDIDATES_CSV)
    df["structure_reference_uniprot_id"] = ""
    at = _app_pointed_at(tmp_path, df, monkeypatch)
    at.run()
    assert not at.exception
    body = " ".join(str(t.value) for t in at.markdown) + " ".join(
        str(t.value) for t in at.text
    )
    assert "No structural family reference" in body


def test_an_unresolvable_structure_offers_the_labelled_placeholder(monkeypatch):
    """The integrity-critical branch: no structure, so nothing is invented."""
    from src.structures import StructureResult

    def unavailable(uniprot_id, allow_network=True):
        return StructureResult(
            uniprot_id=uniprot_id,
            available=False,
            source="unavailable",
            message="Could not retrieve a structure for this reference.",
            entry_url=f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}",
        )

    monkeypatch.setattr("src.structures.get_structure", unavailable)
    at = fresh_app()
    at.run()
    assert not at.exception
    assert any("Could not retrieve" in str(w.value) for w in at.warning)

    placeholder = widget_by_label(at.checkbox, "geometric placeholder")
    placeholder.set_value(True).run()
    assert not at.exception
    # It must be labelled as not being a protein wherever it is shown.
    assert any("not a protein" in str(e.value).lower() for e in at.error)


def test_a_dataset_with_no_scoreable_rows_stops_cleanly(tmp_path, monkeypatch):
    from src.config import CANDIDATES_CSV

    df = pd.read_csv(CANDIDATES_CSV)
    df["plddt_mean"] = None
    at = _app_pointed_at(tmp_path, df, monkeypatch)
    at.run()
    assert not at.exception
    # The empty case is an error, not the per-candidate exclusion warning.
    errors = " ".join(str(e.value) for e in at.error)
    assert "None of the" in errors and "could be scored" in errors
