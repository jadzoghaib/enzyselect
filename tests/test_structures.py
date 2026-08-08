"""Tests for structure resolution and the graceful-degradation chain.

These never touch the network. The cached AlphaFold files are regenerable
artefacts excluded from version control, so tests that need them skip cleanly
on a fresh clone rather than failing.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import STRUCTURE_FAMILIES
from src.structures import (
    STRUCTURE_CACHE_DIR,
    StructureResult,
    get_structure,
    placeholder_backbone_pdb,
    structure_summary,
    viewer_html,
)

CACHED = sorted(STRUCTURE_CACHE_DIR.glob("*.pdb")) if STRUCTURE_CACHE_DIR.is_dir() else []


# --------------------------------------------------------------------------
# Graceful degradation
# --------------------------------------------------------------------------
def test_unknown_accession_offline_degrades_to_links_only():
    result = get_structure("NOT-A-REAL-ACCESSION", allow_network=False)
    assert result.available is False
    assert result.source == "unavailable"
    assert result.pdb_text is None
    assert result.is_real_structure is False
    # It must still hand the user something verifiable, and say what happened.
    assert result.entry_url.startswith("https://alphafold.ebi.ac.uk/entry/")
    assert result.message


def test_every_lookup_carries_the_activity_warning():
    """Integrity guard: a structure must never be presented without it."""
    result = get_structure("NOT-A-REAL-ACCESSION", allow_network=False)
    joined = " ".join(result.warnings).lower()
    assert "does not establish" in joined


def test_offline_lookup_never_raises_for_any_family():
    for spec in STRUCTURE_FAMILIES.values():
        result = get_structure(spec["uniprot"], allow_network=False)
        assert isinstance(result, StructureResult)


@pytest.mark.skipif(not CACHED, reason="no cached structures (run python -m src.structures)")
def test_cached_structure_loads_without_network():
    accession = CACHED[0].name.replace("AF-", "").replace("-F1.pdb", "")
    result = get_structure(accession, allow_network=False)
    assert result.available is True
    assert result.source == "local_cache"
    assert result.is_real_structure is True
    assert result.pdb_text and "ATOM" in result.pdb_text


def test_placeholder_is_never_reported_as_a_real_structure():
    result = StructureResult(uniprot_id="X", available=True, source="placeholder")
    assert result.is_real_structure is False


# --------------------------------------------------------------------------
# The labelled geometric placeholder
# --------------------------------------------------------------------------
def test_placeholder_is_labelled_as_not_a_protein():
    pdb = placeholder_backbone_pdb(20)
    assert "NOT A REAL PROTEIN STRUCTURE" in pdb
    assert "Not a prediction" in pdb


def test_placeholder_has_the_requested_residue_count():
    assert structure_summary(placeholder_backbone_pdb(37))["residues_resolved"] == 37


def test_placeholder_geometry_is_a_helix():
    """Consecutive CA atoms sit ~1.5 A apart along the helix axis."""
    lines = [ln for ln in placeholder_backbone_pdb(5).splitlines() if ln.startswith("ATOM")]
    z = [float(ln[46:54]) for ln in lines]
    steps = [round(b - a, 3) for a, b in pairwise(z)]
    assert len(steps) == 4
    assert all(s == pytest.approx(1.5) for s in steps)


# --------------------------------------------------------------------------
# Coordinate parsing
# --------------------------------------------------------------------------
def test_structure_summary_reads_the_bfactor_column_as_plddt():
    summary = structure_summary(placeholder_backbone_pdb(10))
    assert summary["mean_plddt_from_file"] == pytest.approx(50.0)
    assert summary["pct_residues_plddt_70_plus"] == pytest.approx(0.0)


def test_structure_summary_on_empty_input_is_safe():
    summary = structure_summary("")
    assert summary["residues_resolved"] == 0
    assert summary["mean_plddt_from_file"] is None


@pytest.mark.skipif(not CACHED, reason="no cached structures")
def test_real_model_parses_with_plausible_confidence():
    summary = structure_summary(CACHED[0].read_text(encoding="utf-8"))
    assert summary["residues_resolved"] > 100
    assert 0.0 <= summary["mean_plddt_from_file"] <= 100.0


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------
def test_viewer_html_embeds_coordinates_and_the_library():
    """Guards py3Dmol's private _make_html(); an upgrade that breaks it fails here."""
    html = viewer_html(placeholder_backbone_pdb(30))
    assert html, "viewer_html returned empty - py3Dmol API may have changed"
    assert "3dmolviewer" in html
    assert "ATOM" in html


@pytest.mark.parametrize("bad_input", [None, "", "not a pdb file"])
def test_viewer_html_never_raises_on_degenerate_input(bad_input):
    """The app must never crash on a structure it could not parse.

    py3Dmol is tolerant enough to emit a viewer for junk input, so the
    contract here is 'always returns a string, never raises' rather than
    'returns empty'.
    """
    assert isinstance(viewer_html(bad_input), str)
