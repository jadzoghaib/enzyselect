"""AlphaFold / UniProt integration with graceful degradation.

The application must stay usable with no network, so structure resolution runs
through three tiers, in order:

1. **Local cache** — ``data/structure_cache/*.pdb``, populated by running this
   module as a script. Works fully offline.
2. **Live AlphaFold API** — resolves the current model URL (the model version
   is *not* hardcoded; as of writing AlphaFold DB serves ``_v6`` files, and
   hardcoding ``_v4`` would already be wrong). Successful downloads are cached.
3. **Links only** — if both fail, the UI shows verifiable entry links and says
   plainly that the structure could not be retrieved. It never fabricates one.

A separate, clearly-labelled *geometric placeholder* is available for
demonstrating the viewer offline. It is an idealised poly-alanine helix built
from textbook geometry — it is not a protein, not a prediction, and not
derived from any candidate. It exists so the viewer can be shown working
without pretending to show a structure.

Scientific note carried through this whole module: **a predicted structure is
not evidence of enzyme activity, and pLDDT is confidence in predicted local
geometry, not stability or catalytic performance.**
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import ALPHAFOLD_API_URL, ALPHAFOLD_ENTRY_URL, DATA_DIR

STRUCTURE_CACHE_DIR = DATA_DIR / "structure_cache"
# Structure files run to a few hundred kB and the EBI endpoint is routinely
# slow on a cold request, so the budget is generous and downloads get one
# retry. The UI stays responsive because a successful fetch is cached to disk
# and Streamlit memoizes the result for the session.
REQUEST_TIMEOUT = 20
DOWNLOAD_ATTEMPTS = 2


@dataclass
class StructureResult:
    """Outcome of a structure lookup. Always safe to render."""

    uniprot_id: str
    available: bool = False
    source: str = "unavailable"  # local_cache | alphafold_api | placeholder
    message: str = ""
    pdb_text: str | None = None
    entry_url: str = ""
    model_url: str = ""
    real_mean_plddt: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_real_structure(self) -> bool:
        """True only for an actual downloaded prediction, never a placeholder."""
        return self.available and self.source in {"local_cache", "alphafold_api"}


def _cache_path(uniprot_id: str) -> Path:
    return STRUCTURE_CACHE_DIR / f"AF-{uniprot_id}-F1.pdb"


@lru_cache(maxsize=32)
def fetch_alphafold_metadata(uniprot_id: str) -> dict | None:
    """Resolve model URLs and the real mean pLDDT. None on any failure."""
    try:
        import requests

        response = requests.get(
            ALPHAFOLD_API_URL.format(acc=uniprot_id), timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        if not payload:
            return None
        entry = payload[0]
        return {
            "entry_id": entry.get("entryId", ""),
            "pdb_url": entry.get("pdbUrl", ""),
            "cif_url": entry.get("cifUrl", ""),
            "model_version": entry.get("latestVersion"),
            # REAL: AlphaFold's confidence in its own prediction for the
            # reference protein. Says nothing about activity.
            "mean_plddt": entry.get("globalMetricValue"),
        }
    except Exception:
        return None


def _download(url: str) -> str | None:
    """Download text with one retry. Returns None rather than raising."""
    try:
        import requests
    except ImportError:
        return None

    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200 and response.text.strip():
                return response.text
        except Exception:
            continue
    return None


def get_structure(uniprot_id: str, allow_network: bool = True) -> StructureResult:
    """Resolve a structure through the cache -> API -> links-only chain."""
    result = StructureResult(
        uniprot_id=uniprot_id,
        entry_url=ALPHAFOLD_ENTRY_URL.format(acc=uniprot_id),
    )
    result.warnings.append(
        "A predicted structure is a structural hypothesis. It does not "
        "establish that this or any enzyme degrades PET."
    )

    # Tier 1 — local cache.
    cached = _cache_path(uniprot_id)
    if cached.is_file():
        try:
            text = cached.read_text(encoding="utf-8")
            if text.strip():
                result.available = True
                result.source = "local_cache"
                result.pdb_text = text
                result.message = (
                    f"Loaded the AlphaFold model for {uniprot_id} from the local "
                    "cache (no network required)."
                )
                metadata = fetch_alphafold_metadata(uniprot_id) if allow_network else None
                if metadata:
                    result.model_url = metadata["pdb_url"]
                    result.real_mean_plddt = metadata["mean_plddt"]
                return result
        except OSError:
            pass

    # Tier 2 — live API.
    if allow_network:
        metadata = fetch_alphafold_metadata(uniprot_id)
        if metadata and metadata["pdb_url"]:
            result.model_url = metadata["pdb_url"]
            result.real_mean_plddt = metadata["mean_plddt"]
            downloaded = _download(metadata["pdb_url"])
            if downloaded:
                result.available = True
                result.source = "alphafold_api"
                result.pdb_text = downloaded
                result.message = (
                    f"Retrieved AlphaFold model {metadata['entry_id']} "
                    f"(version {metadata['model_version']}) live from the "
                    "AlphaFold DB API."
                )
                try:
                    STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cached.write_text(downloaded, encoding="utf-8")
                except OSError:
                    pass
                return result

    # Tier 3 — links only. No structure is invented.
    result.message = (
        f"Could not retrieve a structure for {uniprot_id} (no local cache and "
        "the AlphaFold API was unreachable). The verifiable entry link below "
        "still works from a browser."
    )
    return result


# --------------------------------------------------------------------------
# Clearly-labelled geometric placeholder
# --------------------------------------------------------------------------
def placeholder_backbone_pdb(n_residues: int = 60) -> str:
    """An idealised alpha-helix CA trace — a shape, not a protein.

    Built from textbook helix geometry (radius 2.3 A, rise 1.5 A per residue,
    100 degrees per residue). Provided purely so the 3D viewer can be
    demonstrated offline. Every consumer of this function must label it as a
    placeholder; it represents no candidate and no real molecule.
    """
    radius, rise, turn = 2.3, 1.5, math.radians(100.0)
    lines = [
        "REMARK  GEOMETRIC PLACEHOLDER - NOT A REAL PROTEIN STRUCTURE",
        "REMARK  Idealised poly-alanine alpha-helix CA trace.",
        "REMARK  Not a prediction. Not derived from any enzyme candidate.",
    ]
    for i in range(1, int(n_residues) + 1):
        angle = turn * (i - 1)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = rise * (i - 1)
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 50.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def viewer_html(pdb_text: str, height: int = 420, style: str = "cartoon") -> str:
    """Standalone py3Dmol HTML for embedding in Streamlit.

    Note the honest limitation: py3Dmol loads the 3Dmol.js library from a CDN
    inside the iframe, so the *viewer* needs internet even when the structure
    itself came from the local cache. When it cannot load, the app falls back
    to links and metadata rather than showing an empty box.
    """
    try:
        import py3Dmol

        view = py3Dmol.view(width="100%", height=height)
        view.addModel(pdb_text, "pdb")
        if style == "cartoon":
            # Colour by the B-factor column, which in AlphaFold PDB files
            # carries per-residue pLDDT. Confidence in geometry only.
            view.setStyle(
                {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",
                                             "min": 50, "max": 90}}}
            )
        else:
            view.setStyle({"sphere": {"radius": 0.6}, "stick": {"radius": 0.15}})
        view.zoomTo()
        return view._make_html()
    except Exception:
        return ""


def structure_summary(pdb_text: str) -> dict:
    """Parse a few descriptive facts straight from the coordinate file."""
    ca_count = 0
    plddt_values = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca_count += 1
            # A malformed B-factor column costs us one confidence value, not
            # the whole parse.
            with contextlib.suppress(ValueError):
                plddt_values.append(float(line[60:66]))
    mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else None
    confident = (
        sum(1 for v in plddt_values if v >= 70) / len(plddt_values) * 100.0
        if plddt_values
        else None
    )
    return {
        "residues_resolved": ca_count,
        "mean_plddt_from_file": mean_plddt,
        "pct_residues_plddt_70_plus": confident,
    }


def prefetch_all() -> None:
    """Populate the local cache so the demo runs offline. Run as a script."""
    from .config import STRUCTURE_FAMILIES

    STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for family, spec in STRUCTURE_FAMILIES.items():
        acc = str(spec["uniprot"])
        result = get_structure(acc, allow_network=True)
        status = "cached" if result.is_real_structure else "FAILED"
        size = len(result.pdb_text or "")
        print(f"  {family:<36} {acc:<12} {status:<8} {size:>8,} bytes")


if __name__ == "__main__":  # pragma: no cover
    print("Prefetching AlphaFold reference structures into the local cache:")
    prefetch_all()
