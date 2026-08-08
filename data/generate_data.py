"""Generate the EnzySelect synthetic candidate dataset.

Run:  python data/generate_data.py

What this script produces
-------------------------
1. ``data/candidates.csv`` — 50 SYNTHETIC enzyme candidates. These candidates
   do not exist. Every performance-like value is drawn from a random process
   with a fixed seed. No row makes an experimental claim about a real enzyme.

2. ``data/structure_references.csv`` — a small table of REAL, verifiable
   structural anchors (UniProt accessions + AlphaFold DB entries), fetched
   live from the UniProt and AlphaFold REST APIs where possible.

The separation between the two files is the point: real metadata lives in one
file, synthetic estimates in the other, and the candidate table references the
anchor only as a *structural family reference*, never as its own structure.

Offline behaviour: if the APIs are unreachable the script falls back to the
constants in ``src/config.py`` and records that it did so, so the dataset is
always reproducible without network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ALPHAFOLD_API_URL,
    ALPHAFOLD_ENTRY_URL,
    CANDIDATES_CSV,
    EVIDENCE_LEVELS,
    EXPERIMENTAL_STATUSES,
    N_CANDIDATES,
    RANDOM_SEED,
    REFERENCES_CSV,
    SOURCE_ENVIRONMENTS,
    STRUCTURE_FAMILIES,
    UNIPROT_API_URL,
)

TIMEOUT = 12


# --------------------------------------------------------------------------
# Step 1 — resolve the REAL structural anchors
# --------------------------------------------------------------------------
def _get_json(url: str):
    """GET and parse JSON, returning None on any failure. Never raises."""
    try:
        import requests

        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:  # network down, DNS, SSL, malformed JSON, no requests
        return None


def resolve_reference(family: str, spec: dict) -> dict:
    """Look up one real anchor. Falls back to config constants when offline."""
    acc = spec["uniprot"]
    record = {
        "structure_family": family,
        "family_code": spec["code"],
        "reference_uniprot_id": acc,
        "reference_protein_name": spec["ref_name"],
        "reference_organism": spec["ref_organism"],
        "reference_sequence_length": spec["ref_length"],
        "uniprot_entry_url": f"https://www.uniprot.org/uniprotkb/{acc}/entry",
        "alphafold_db_url": ALPHAFOLD_ENTRY_URL.format(acc=acc),
        "alphafold_model_pdb_url": "",
        "alphafold_model_cif_url": "",
        "reference_plddt_mean_real": np.nan,
        "metadata_source": "offline_fallback_from_config",
    }

    uni = _get_json(UNIPROT_API_URL.format(acc=acc))
    if uni:
        try:
            record["reference_protein_name"] = (
                uni["proteinDescription"]["recommendedName"]["fullName"]["value"]
            )
            record["reference_organism"] = uni["organism"]["scientificName"]
            record["reference_sequence_length"] = int(uni["sequence"]["length"])
            record["metadata_source"] = "uniprot_rest_api"
        except (KeyError, TypeError, ValueError):
            pass

    af = _get_json(ALPHAFOLD_API_URL.format(acc=acc))
    if af:
        try:
            entry = af[0]
            record["alphafold_model_pdb_url"] = entry.get("pdbUrl", "")
            record["alphafold_model_cif_url"] = entry.get("cifUrl", "")
            # This pLDDT is REAL: it is AlphaFold's own confidence in its
            # prediction for the reference protein. It says nothing about
            # activity, and nothing at all about the synthetic candidates.
            record["reference_plddt_mean_real"] = entry.get("globalMetricValue", np.nan)
            if record["metadata_source"] == "uniprot_rest_api":
                record["metadata_source"] = "uniprot_rest_api + alphafold_api"
            else:
                record["metadata_source"] = "alphafold_api"
        except (IndexError, KeyError, TypeError):
            pass

    return record


# Fields that describe a *live lookup*. When the APIs are unreachable these
# must be carried over from the previous verified run rather than blanked:
# overwriting them would silently destroy real metadata (and, downstream, every
# candidate's structure URL) as a side effect of being offline.
PRESERVED_WHEN_OFFLINE = [
    "reference_protein_name",
    "reference_organism",
    "reference_sequence_length",
    "alphafold_model_pdb_url",
    "alphafold_model_cif_url",
    "reference_plddt_mean_real",
    "metadata_source",
    "verified_on",
]


def build_reference_table() -> pd.DataFrame:
    rows = [resolve_reference(name, spec) for name, spec in STRUCTURE_FAMILIES.items()]
    df = pd.DataFrame(rows)
    df["verified_on"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    df["data_provenance"] = "real_public_database_record"

    offline = df["metadata_source"] == "offline_fallback_from_config"
    if not offline.any():
        return df

    restored = 0
    if REFERENCES_CSV.is_file():
        prior = pd.read_csv(REFERENCES_CSV).drop_duplicates(
            subset="reference_uniprot_id"
        ).set_index("reference_uniprot_id")
        for idx in df.index[offline]:
            accession = df.at[idx, "reference_uniprot_id"]
            if accession not in prior.index:
                continue
            for column in PRESERVED_WHEN_OFFLINE:
                if column in prior.columns and pd.notna(prior.at[accession, column]):
                    df.at[idx, column] = prior.at[accession, column]
            restored += 1

    unresolved = int(offline.sum()) - restored
    print(
        f"  WARNING: {int(offline.sum())} anchor(s) could not be verified "
        f"(APIs unreachable). Preserved {restored} from the previous verified "
        f"run; {unresolved} have no prior record."
    )
    if unresolved:
        print(
            "  Those anchors will have an empty model URL, which will also "
            "empty predicted_structure_url for their candidates. Re-run with "
            "network access to restore them."
        )
    return df


# --------------------------------------------------------------------------
# Step 2 — generate the SYNTHETIC candidates
# --------------------------------------------------------------------------
def _weighted_choice(rng: np.random.Generator, mapping: dict, n: int) -> np.ndarray:
    keys = list(mapping)
    probs = np.array([mapping[k]["share"] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_candidates(
    n: int = N_CANDIDATES, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the synthetic candidate table.

    Two latent factors drive the correlations so the dataset contains genuine
    trade-offs rather than one dominant candidate:

    ``z_quality``     raises pLDDT, catalytic confidence, degradation rate and
                      evidence level together.
    ``z_difficulty``  raises expression difficulty and production cost.

    Production cost also loads *positively* on ``z_quality`` (+0.30), which is
    what stops the ranking from being trivial: the strongest performers tend to
    be the most expensive to make, so the weighting choice actually matters.
    """
    rng = np.random.default_rng(seed)

    families = _weighted_choice(rng, STRUCTURE_FAMILIES, n)
    environments = _weighted_choice(rng, SOURCE_ENVIRONMENTS, n)

    z_quality = rng.normal(0.0, 1.0, n)
    z_difficulty = rng.normal(0.0, 1.0, n)

    fam_temp = np.array([STRUCTURE_FAMILIES[f]["temp_mu"] for f in families])
    fam_ph = np.array([STRUCTURE_FAMILIES[f]["ph_mu"] for f in families])
    fam_len_mu = np.array([STRUCTURE_FAMILIES[f]["len_mu"] for f in families])
    fam_len_sd = np.array([STRUCTURE_FAMILIES[f]["len_sd"] for f in families])
    fam_rate = np.array([STRUCTURE_FAMILIES[f]["rate_shift"] for f in families])
    fam_code = np.array([STRUCTURE_FAMILIES[f]["code"] for f in families])

    env_temp_shift = np.array([SOURCE_ENVIRONMENTS[e]["temp_shift"] for e in environments])
    env_salt_mu = np.array([SOURCE_ENVIRONMENTS[e]["salt_mu"] for e in environments])

    # --- structural confidence (synthetic) --------------------------------
    plddt = np.clip(82.0 + 6.5 * z_quality + rng.normal(0, 4.0, n), 55.0, 97.0)
    catalytic = np.clip(
        _logistic(0.55 + 0.80 * z_quality + rng.normal(0, 0.45, n)), 0.28, 0.98
    )

    # --- process characteristics (synthetic) ------------------------------
    temp_opt = np.clip(fam_temp + env_temp_shift + rng.normal(0, 4.5, n), 28.0, 88.0)
    ph_opt = np.clip(fam_ph + rng.normal(0, 0.7, n), 5.5, 10.5)
    salinity = np.clip(
        env_salt_mu + rng.normal(0, 0.35 * env_salt_mu + 6.0, n), 0.0, 160.0
    )

    # --- performance (synthetic, log-normal) ------------------------------
    log_rate = -0.35 + 0.42 * z_quality + fam_rate + rng.normal(0, 0.26, n)
    degradation_rate = np.clip(10.0**log_rate, 0.02, 6.0)

    # --- production feasibility (synthetic) -------------------------------
    seq_len = np.round(rng.normal(fam_len_mu, fam_len_sd)).astype(int)
    difficulty = np.clip(
        np.round(3.0 + 1.05 * z_difficulty + 0.35 * z_quality), 1, 5
    ).astype(int)
    cost = np.clip(
        190.0
        * np.exp(
            0.45 * z_difficulty + 0.30 * z_quality + 0.0016 * (seq_len - 290.0)
        )
        * rng.lognormal(0.0, 0.16, n),
        60.0,
        1500.0,
    )

    # --- evidence and status (synthetic) ----------------------------------
    evidence_latent = 0.9 * z_quality + rng.normal(0, 0.8, n)
    evidence_level = pd.qcut(
        evidence_latent, q=[0, 0.30, 0.55, 0.75, 0.90, 1.0], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    status_idx = np.clip(
        np.round(evidence_level * 0.75 + rng.normal(0, 0.45, n)), 0, 3
    ).astype(int)

    refs = build_reference_table().set_index("structure_family")

    df = pd.DataFrame(
        {
            "candidate_id": [f"ENZ-SYN-{i:03d}" for i in range(1, n + 1)],
            "enzyme_name": [
                f"SYN-{code}-{i:03d}"
                for i, code in zip(range(1, n + 1), fam_code, strict=True)
            ],
            "structure_family": families,
            "organism": environments,
            "sequence_length": seq_len,
            # Synthetic candidates have no accession. Deliberately empty.
            "uniprot_id": "",
            "plddt_mean": np.round(plddt, 1),
            "catalytic_site_confidence": np.round(catalytic, 3),
            "estimated_temperature_optimum": np.round(temp_opt, 1),
            "estimated_ph_optimum": np.round(ph_opt, 2),
            "estimated_salinity_tolerance": np.round(salinity, 1),
            "synthetic_degradation_rate": np.round(degradation_rate, 3),
            "estimated_expression_difficulty": difficulty,
            "estimated_production_cost": np.round(cost, 0),
            "literature_evidence_level": evidence_level,
            "literature_evidence_label": [EVIDENCE_LEVELS[v] for v in evidence_level],
            "experimental_status": [EXPERIMENTAL_STATUSES[i] for i in status_idx],
            "data_provenance": "synthetic",
        }
    )

    # Family-reference structure links: REAL entries for a characterized
    # representative of the family — never a structure of the candidate.
    df["structure_reference_uniprot_id"] = [
        refs.loc[f, "reference_uniprot_id"] for f in families
    ]
    df["structure_reference_name"] = [
        refs.loc[f, "reference_protein_name"] for f in families
    ]
    df["structure_reference_organism"] = [
        refs.loc[f, "reference_organism"] for f in families
    ]
    df["alphafold_db_url"] = [refs.loc[f, "alphafold_db_url"] for f in families]
    df["predicted_structure_url"] = [
        refs.loc[f, "alphafold_model_pdb_url"] for f in families
    ]
    df["structure_link_type"] = "family_reference_not_candidate_specific"

    return df, refs.reset_index()


def main() -> None:
    candidates, references = generate_candidates()

    CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATES_CSV, index=False)
    references.to_csv(REFERENCES_CSV, index=False)

    print(f"Wrote {len(candidates)} synthetic candidates -> {CANDIDATES_CSV}")
    print(f"Wrote {len(references)} real structure references -> {REFERENCES_CSV}")
    print("\nMetadata sources for the real anchors:")
    for _, row in references.iterrows():
        plddt = row["reference_plddt_mean_real"]
        plddt_txt = "n/a" if pd.isna(plddt) else f"{plddt:.2f}"
        print(
            f"  {row['reference_uniprot_id']:<12} {row['metadata_source']:<32} "
            f"real mean pLDDT={plddt_txt}"
        )
    print("\nSanity checks on the synthetic table:")
    print(candidates[
        [
            "estimated_temperature_optimum",
            "estimated_ph_optimum",
            "estimated_salinity_tolerance",
            "synthetic_degradation_rate",
            "plddt_mean",
            "estimated_production_cost",
        ]
    ].describe().round(2).to_string())


if __name__ == "__main__":
    main()
