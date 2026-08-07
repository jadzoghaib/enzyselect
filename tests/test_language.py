"""Guards on the language the application is allowed to use.

Scientific-integrity wording is easy to erode one convenient phrase at a time.
These tests make the overclaiming vocabulary a build failure rather than a
matter of memory.

Scope note: ``src/config.py`` and ``README.md`` are excluded because they
*quote* the banned list in order to document the policy. Every module that
produces user-facing strings is in scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BANNED_TERMS, SCIENTIFIC_INTEGRITY_NOTES  # noqa: E402

SCANNED_FILES = [
    ROOT / "app.py",
    ROOT / "src" / "scoring.py",
    ROOT / "src" / "economics.py",
    ROOT / "src" / "visualizations.py",
    ROOT / "src" / "structures.py",
    ROOT / "data" / "generate_data.py",
]


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda p: p.name)
def test_no_overclaiming_vocabulary(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    hits = [term for term in BANNED_TERMS if term.lower() in text]
    assert not hits, f"{path.name} uses banned overclaiming language: {hits}"


def test_dataset_is_labelled_synthetic():
    df = pd.read_csv(ROOT / "data" / "candidates.csv")
    assert (df["data_provenance"] == "synthetic").all()


def test_synthetic_candidates_carry_no_uniprot_accession():
    """A synthetic candidate must never masquerade as a real database entry."""
    df = pd.read_csv(ROOT / "data" / "candidates.csv")
    assert df["uniprot_id"].isna().all() or (df["uniprot_id"].fillna("") == "").all()


def test_structure_links_are_labelled_as_family_references():
    df = pd.read_csv(ROOT / "data" / "candidates.csv")
    assert (df["structure_link_type"] == "family_reference_not_candidate_specific").all()


def test_reference_table_only_holds_real_records():
    refs = pd.read_csv(ROOT / "data" / "structure_references.csv")
    assert (refs["data_provenance"] == "real_public_database_record").all()
    # Real anchors must not carry any synthetic performance column.
    forbidden = {"synthetic_degradation_rate", "catalytic_site_confidence",
                 "estimated_production_cost", "overall_score"}
    assert not forbidden & set(refs.columns)


def test_integrity_notes_cover_the_required_warnings():
    joined = " ".join(SCIENTIFIC_INTEGRITY_NOTES).lower()
    for required in ["alphafold", "plddt", "synthetic", "experimental validation",
                     "regulatory", "safely degrade plastic"]:
        assert required in joined, f"missing required warning about: {required}"


def test_app_surfaces_the_headline_disclaimer():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "HEADLINE_DISCLAIMER" in text
    assert "SCIENTIFIC_INTEGRITY_NOTES" in text
