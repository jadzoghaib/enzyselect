"""Tests for the live-network branch of structure resolution.

Nothing here touches the network. ``structures.py`` imports ``requests``
*inside* its functions, so a fake module dropped into ``sys.modules`` is
picked up at call time — which lets the whole three-tier fallback chain be
exercised deterministically, including the failure paths that are impossible
to trigger reliably against a real API.

This is the code that backs the README's graceful-degradation claim. Before
these tests only the offline branch was covered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import structures
from src.structures import (
    fetch_alphafold_metadata,
    get_structure,
    placeholder_backbone_pdb,
    prefetch_all,
    viewer_html,
)

PDB_BODY = placeholder_backbone_pdb(12)
API_PAYLOAD = [
    {
        "entryId": "AF-TEST123-F1",
        "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-TEST123-F1-model_v6.pdb",
        "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-TEST123-F1-model_v6.cif",
        "latestVersion": 6,
        "globalMetricValue": 91.5,
    }
]


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeRequests:
    """Stands in for the ``requests`` module and records every call."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        outcome = self._handler(url, len(self.calls))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Clear the metadata memo and redirect the cache into a temp directory.

    Without the redirect these tests would write into the repository's real
    structure cache; without the memo clear, results would leak between tests.
    """
    fetch_alphafold_metadata.cache_clear()
    monkeypatch.setattr(structures, "STRUCTURE_CACHE_DIR", tmp_path / "cache")
    yield
    fetch_alphafold_metadata.cache_clear()


def install(monkeypatch, handler) -> FakeRequests:
    fake = FakeRequests(handler)
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def serve_everything(url, attempt):
    if "/api/prediction/" in url:
        return FakeResponse(payload=API_PAYLOAD)
    return FakeResponse(text=PDB_BODY)


# --------------------------------------------------------------------------
# Metadata lookup
# --------------------------------------------------------------------------
def test_metadata_is_parsed_from_the_api(monkeypatch):
    install(monkeypatch, serve_everything)
    meta = fetch_alphafold_metadata("TEST123")
    assert meta["entry_id"] == "AF-TEST123-F1"
    assert meta["model_version"] == 6
    assert meta["mean_plddt"] == 91.5
    assert meta["pdb_url"].endswith("-model_v6.pdb")


def test_metadata_does_not_hardcode_a_model_version(monkeypatch):
    """The URL must come from the API. Assuming _v4 was already wrong."""
    install(monkeypatch, serve_everything)
    assert "_v4" not in fetch_alphafold_metadata("TEST123")["pdb_url"]


@pytest.mark.parametrize(
    "handler,label",
    [
        (lambda url, n: FakeResponse(status_code=404), "not found"),
        (lambda url, n: FakeResponse(payload=[]), "empty payload"),
        (lambda url, n: FakeResponse(payload=None), "malformed json"),
        (lambda url, n: TimeoutError("timed out"), "timeout"),
        (lambda url, n: ConnectionError("dns"), "connection refused"),
    ],
)
def test_metadata_returns_none_on_any_failure(monkeypatch, handler, label):
    install(monkeypatch, handler)
    assert fetch_alphafold_metadata("TEST123") is None, label


# --------------------------------------------------------------------------
# Download, including the retry
# --------------------------------------------------------------------------
def test_download_retries_once_before_giving_up(monkeypatch):
    fake = install(monkeypatch, lambda url, n: FakeResponse(status_code=500))
    assert structures._download("http://example/x.pdb") is None
    assert len(fake.calls) == structures.DOWNLOAD_ATTEMPTS == 2


def test_download_recovers_on_the_second_attempt(monkeypatch):
    def flaky(url, attempt):
        return TimeoutError("first attempt dies") if attempt == 1 else FakeResponse(text=PDB_BODY)

    fake = install(monkeypatch, flaky)
    assert structures._download("http://example/x.pdb") == PDB_BODY
    assert len(fake.calls) == 2


def test_download_rejects_a_blank_body(monkeypatch):
    install(monkeypatch, lambda url, n: FakeResponse(text="   \n  "))
    assert structures._download("http://example/x.pdb") is None


def test_download_without_requests_installed_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name))
        if name == "requests"
        else __import__(name, *a, **k),
    )
    assert structures._download("http://example/x.pdb") is None


# --------------------------------------------------------------------------
# The three-tier chain
# --------------------------------------------------------------------------
def test_tier_two_fetches_live_and_writes_the_cache(monkeypatch):
    install(monkeypatch, serve_everything)
    result = get_structure("TEST123", allow_network=True)
    assert result.available and result.is_real_structure
    assert result.source == "alphafold_api"
    assert result.real_mean_plddt == 91.5
    assert "version 6" in result.message
    # The download must have been persisted for the next, offline run.
    assert structures._cache_path("TEST123").read_text(encoding="utf-8") == PDB_BODY


def test_a_cached_structure_is_preferred_over_the_network(monkeypatch):
    install(monkeypatch, serve_everything)
    get_structure("TEST123", allow_network=True)          # populate the cache
    fake = install(monkeypatch, serve_everything)          # fresh call recorder
    fetch_alphafold_metadata.cache_clear()
    result = get_structure("TEST123", allow_network=True)
    assert result.source == "local_cache"
    # Metadata may be re-checked, but the coordinates must not be re-downloaded.
    assert not [u for u in fake.calls if "/files/" in u]


def test_tier_three_when_the_model_cannot_be_downloaded(monkeypatch):
    def metadata_only(url, attempt):
        if "/api/prediction/" in url:
            return FakeResponse(payload=API_PAYLOAD)
        return FakeResponse(status_code=503)

    install(monkeypatch, metadata_only)
    result = get_structure("TEST123", allow_network=True)
    assert result.available is False
    assert result.is_real_structure is False
    assert result.pdb_text is None
    # It still hands back something verifiable rather than nothing.
    assert result.model_url.endswith(".pdb")
    assert result.entry_url.endswith("TEST123")
    assert "Could not retrieve" in result.message


def test_network_is_never_touched_when_disallowed(monkeypatch):
    fake = install(monkeypatch, serve_everything)
    result = get_structure("TEST123", allow_network=False)
    assert result.available is False
    assert fake.calls == []


def test_an_unwritable_cache_does_not_break_the_fetch(monkeypatch):
    """A read-only cache directory must degrade, not raise."""
    install(monkeypatch, serve_everything)
    monkeypatch.setattr(
        Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )
    result = get_structure("TEST123", allow_network=True)
    assert result.available is True
    assert result.source == "alphafold_api"


def test_an_unreadable_cache_file_falls_through_to_the_network(monkeypatch):
    install(monkeypatch, serve_everything)
    cache = structures._cache_path("TEST123")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("   ", encoding="utf-8")  # present but empty
    result = get_structure("TEST123", allow_network=True)
    assert result.source == "alphafold_api"


# --------------------------------------------------------------------------
# Prefetch helper and the viewer's alternate paths
# --------------------------------------------------------------------------
def test_prefetch_caches_every_family(monkeypatch, capsys):
    install(monkeypatch, serve_everything)
    prefetch_all()
    out = capsys.readouterr().out
    assert "cached" in out
    assert "FAILED" not in out
    assert len(list(structures.STRUCTURE_CACHE_DIR.glob("*.pdb"))) == 5


def test_prefetch_reports_failures_without_raising(monkeypatch, capsys):
    install(monkeypatch, lambda url, n: ConnectionError("offline"))
    prefetch_all()
    assert "FAILED" in capsys.readouterr().out


def test_viewer_supports_the_trace_style():
    html = viewer_html(PDB_BODY, style="trace")
    assert "3dmolviewer" in html
    assert "sphere" in html


def test_viewer_returns_empty_when_py3dmol_fails(monkeypatch):
    import py3Dmol

    monkeypatch.setattr(
        py3Dmol, "view", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no webgl"))
    )
    assert viewer_html(PDB_BODY) == ""
