# EnzySelect

**AI-assisted enzyme candidate prioritization for PET plastic degradation.**

> ## ⚠️ Educational prototype — demonstration only
>
> EnzySelect is a teaching and portfolio project. It is **not** a validated
> scientific, medical, environmental or industrial decision-making system.
>
> - The candidate dataset is **synthetic**. The 50 candidates do not exist and
>   no value in the table is a measurement of any real enzyme.
> - Nothing here is evidence that any enzyme degrades PET, in a laboratory or
>   anywhere else.
> - AlphaFold structure predictions **do not establish enzyme activity**.
> - Every output is a *prioritization signal* that would require experimental
>   validation before it meant anything.
> - This application provides no regulatory, medical, environmental or safety
>   approval of any kind.

---

## 1. Project objective

Show how a decision-support tool could help an R&D team decide **which enzyme
candidates to put into the laboratory first**, and — just as importantly — show
how such a tool should communicate its own uncertainty.

The engineering goal is a transparent, reproducible, fully local application:
no paid API, no private key, no cloud account, no laboratory data.

## 2. Business problem

Screening enzyme candidates is slow and expensive. A laboratory that can run a
handful of assays per cycle, at a few thousand euro each, cannot exhaustively
test a large candidate pool. Somebody has to choose an order.

That choice is usually made informally. EnzySelect makes it **explicit,
reproducible and arguable**: the user states the process conditions and what
they care about, and the tool produces a ranking that can be inspected,
challenged and re-run under different assumptions.

The honest framing matters here. The tool does not find good enzymes. It
orders a queue according to preferences the user declared, using data whose
quality it cannot vouch for.

## 3. User journey

1. **Set process requirements** in the sidebar — operating temperature, pH,
   salinity, testing budget, maximum number of candidates to test, and the size
   of the original screening pool the baseline cost is measured against.
2. **Tune the scoring weights** to reflect what actually matters for the
   programme (raw performance? manufacturability? evidence?).
3. **Read the executive summary** — how many candidates were assessed, how many
   fit the budget, the top candidate, the illustrative cost avoided.
4. **Inspect the ranked table**, colour-banded by priority and always paired
   with a written recommended action.
5. **Open a candidate deep dive** — metadata, score breakdown, radar and bar
   comparison against the portfolio, a real AlphaFold reference structure for
   its structural family, and a plain-language explanation of the score.
6. **Run a scenario** — change temperature, pH, salinity, cost per test or the
   candidate cap, and see how far the ranking moves. Large movement is a
   warning about the method, not a discovery about the enzymes.
7. **Review the illustrative economics** and its sensitivity to the cost
   assumption.
8. **Download** the shortlist, the score breakdown, the assumptions and the
   business case as CSV.

## 4. Technical architecture

```
enzyselect/
├── app.py                       Streamlit dashboard (7 sections, 6 tabs)
├── requirements.txt             Runtime dependencies only
├── requirements-dev.txt         Adds pytest, coverage, ruff, mypy
├── ruff.toml                    Correctness-weighted lint configuration
├── Dockerfile
├── README.md
├── .streamlit/config.toml       Pins the light theme the palette was validated against
├── data/
│   ├── generate_data.py         Builds the synthetic dataset (fixed seed)
│   ├── candidates.csv           50 SYNTHETIC candidates
│   ├── structure_references.csv 5 REAL, API-verified structural anchors
│   └── structure_cache/         Cached AlphaFold PDB files (offline fallback)
├── src/
│   ├── config.py                Disclaimers, schema, weights, palette — one source of truth
│   ├── scoring.py               Fit functions, components, ranking, explanations
│   ├── economics.py             Illustrative cost and time arithmetic
│   ├── structures.py            AlphaFold/UniProt integration + graceful degradation
│   └── visualizations.py        Plotly figures
├── docs/screenshots/            Placeholder — no screenshots captured yet
└── tests/                       93 tests: scoring, economics, structures,
                                 figures, and integrity-language guards
```

**Data flow**

```
candidates.csv ──► score_candidates() ──► select_shortlist() ──► ranked view
                        ▲                        │
   sidebar settings ────┘                        ├──► compute_economics() ──► economics view
   (conditions, weights, tolerances)             └──► rank_movement() ─────► scenario view

structure_references.csv ──► structures.get_structure()
                                  │
      local cache ──► AlphaFold API ──► links-only  (three-tier fallback)
```

**Design decisions worth noting**

- `src/config.py` is the single source of truth for every disclaimer, so the
  wording cannot drift between the UI, the exports and this README.
- The scoring engine is pure pandas/NumPy with no Streamlit import, so it is
  testable and reusable outside the dashboard.
- Chart colours come from a validated palette and are checked against
  colour-vision-deficiency separation thresholds. Colour never carries meaning
  alone: every band has a written label and every chart has a table twin.

## 5. Data schema

`data/candidates.csv` — **synthetic**, 50 rows:

| Column | Type | Provenance | Meaning |
|---|---|---|---|
| `candidate_id` | str | synthetic | `ENZ-SYN-001` … |
| `enzyme_name` | str | synthetic | `SYN-<family>-<n>` — deliberately not a real enzyme name |
| `structure_family` | str | modelling choice | One of five structural families |
| `organism` | str | synthetic | Simulated metagenomic provenance, not a species claim |
| `sequence_length` | int | synthetic | Residues |
| `uniprot_id` | str | **empty by design** | Synthetic candidates have no accession |
| `plddt_mean` | float | **synthetic** | Illustrative structural-confidence value |
| `catalytic_site_confidence` | float 0–1 | **synthetic** | Illustrative |
| `estimated_temperature_optimum` | float °C | **synthetic** | |
| `estimated_ph_optimum` | float | **synthetic** | |
| `estimated_salinity_tolerance` | float g/L NaCl | **synthetic** | |
| `synthetic_degradation_rate` | float | **synthetic** | Arbitrary demo units (mg PET / mg enzyme / h) |
| `estimated_expression_difficulty` | int 1–5 | **synthetic** | |
| `estimated_production_cost` | float EUR/g | **synthetic** | |
| `literature_evidence_level` | int 0–4 | **synthetic** | Simulated evidence tier |
| `experimental_status` | str | **synthetic** | `in_silico_only` … `pilot_evaluated` |
| `structure_reference_uniprot_id` | str | **real** | Family anchor accession |
| `alphafold_db_url` / `predicted_structure_url` | str | **real** | Links for the *family reference* |
| `structure_link_type` | str | — | Always `family_reference_not_candidate_specific` |
| `data_provenance` | str | — | Always `synthetic` |

`data/structure_references.csv` — **real** public records, retrieved live from
the UniProt and AlphaFold REST APIs and verified on 2026-08-07:

| Family | UniProt | Protein name | Organism |
|---|---|---|---|
| PETase-like | `A0A0K8P6T7` | Poly(ethylene terephthalate) hydrolase | *Piscinibacter sakaiensis* |
| Leaf-branch-compost-cutinase-like | `G9BY57` | Leaf-branch compost cutinase | Unknown prokaryotic organism |
| Thermobifida-cutinase-like | `Q6A0I4` | Cutinase cut2 | *Thermobifida fusca* |
| Fungal-cutinase-like | `P00590` | Cutinase 1 | *Fusarium vanettenii* |
| MHET-hydrolase-like | `A0A0K8P8E7` | Mono(2-hydroxyethyl) terephthalate hydrolase | *Piscinibacter sakaiensis* |

Sources: [UniProt](https://www.uniprot.org/) · [AlphaFold Protein Structure
Database](https://alphafold.ebi.ac.uk/). These rows carry **no synthetic
performance values** — a test enforces that.

> The two files are deliberately separate. Attaching an invented degradation
> rate to a row labelled with a real enzyme's name would be fabricating an
> experimental claim, so the synthetic candidates reference a family anchor for
> structure only and never borrow its identity.

## 6. Scoring formula

```
score = 100 × Σ wᵢ · sᵢ        where  Σ wᵢ = 1  and  sᵢ ∈ [0, 1]
```

| Component | Default weight | Definition |
|---|---|---|
| Process-condition fit | 30% | 0.45·temp + 0.35·pH + 0.20·salinity |
| Synthetic degradation performance | 25% | log₁₀(rate), min-max normalized across the pool |
| Catalytic-site confidence | 15% | Synthetic 0–1 confidence, used directly |
| Structural confidence (pLDDT) | 10% | (pLDDT − 50) / 50, clipped to [0, 1] |
| Production feasibility | 10% | ½·(1 − normalized log cost) + ½·(1 − (difficulty−1)/4) |
| Evidence level | 10% | Synthetic tier / 4 |

Fit functions:

- **Temperature and pH** — Gaussian: `exp(−½ · ((value − target) / tolerance)²)`.
  At one tolerance from the target the fit is ≈ 0.61.
- **Salinity** — threshold: full credit once tolerance meets the requirement,
  linear falloff below it, **no bonus for excess** (tolerating more salt than
  the process contains is not an advantage).

All weights are renormalized to sum to 1, so only their relative size matters.
Setting every weight to zero falls back to uniform weighting rather than
dividing by zero.

**The score is a decision-support heuristic, not a validated biological
model.** Changing the weights changes the ranking — that sensitivity is the
point of the tool and also its central limitation.

### Known limitations of the method

- **Symmetric temperature/pH penalties.** Real enzymes usually lose activity far
  faster above their optimum than below it. The symmetric Gaussian flatters
  candidates that would be running too hot.
- **Pool-relative normalization.** Performance and the cost half of feasibility
  are min-max normalized across the loaded set, so adding or removing
  candidates changes everyone else's score.
- **Additive independence.** A linear weighted sum cannot express "high
  activity is worthless if the protein will not express".
- **Stated, not fitted, weights.** There is no ground truth in this dataset to
  fit them against, so no accuracy claim of any kind can be made.

## 7. Data limitations

- All performance-like values are drawn from a seeded random process with
  hand-chosen correlations. They are realistic in *shape* only.
- The correlation structure (better candidates cost more to produce; marine
  isolates tolerate more salt; thermophilic families run hotter) was imposed to
  create genuine trade-offs, not learned from data.
- The `literature_evidence_level` and `experimental_status` fields are
  simulated. They do not correspond to any publication.
- Because the dataset is synthetic, the ranking cannot be evaluated for
  accuracy. There is nothing to be right or wrong about.

### Handling incomplete data

A candidate missing any value the scoring model needs — blank, non-numeric or
infinite — is **excluded from the ranking and named in a warning**, not scored
zero and not ranked last. A gap in the data is a fact about the dataset, not a
judgement about the enzyme, and the two must not be allowed to look alike. The
excluded rows are left out of the economics as well, so the shortlist and the
cost figures always describe the same set of candidates.

Duplicate `candidate_id` values are rejected outright at load time: they would
otherwise fan out the scenario comparison into a partial cross product, which
produces a table that looks plausible and is wrong.

## 8. What pLDDT means — and what it does not

pLDDT is AlphaFold's **per-residue confidence in its own predicted local
structure**, on a 0–100 scale. Roughly: > 90 very high confidence, 70–90
confident, 50–70 low, < 50 very low and often indicative of disorder.

**pLDDT is not:**

- a measure of thermal or chemical **stability**;
- a measure of **catalytic activity** or reaction rate;
- a measure of **expression yield** or manufacturability;
- a measure of **industrial suitability**;
- evidence that the protein performs any particular function.

A confidently predicted fold is a structural hypothesis. EnzySelect gives it
10% of the score as a weak positive signal — the model at least knows what
shape it is talking about — and the interface repeats this caveat everywhere
pLDDT appears. The candidate `plddt_mean` values are themselves synthetic; only
the reference-structure pLDDT figures are real AlphaFold outputs.

## 9. Local setup

Requires Python 3.10 or newer.

```bash
git clone <your-fork-url> enzyselect
cd enzyselect

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/generate_data.py     # regenerate the synthetic dataset (optional)
python -m src.structures         # cache reference structures for offline use (optional)

streamlit run app.py
```

The app opens at <http://localhost:8501>.

No API key, no account and no network access are required. With the sidebar
toggle **Allow external API calls** switched off, the app uses only locally
cached structures; the ranking never depends on network access at all.

Run the tests and quality checks with:

```bash
pip install -r requirements-dev.txt

python -m pytest tests -q          # 93 tests
python -m ruff check .             # lint
python -m mypy src app.py --ignore-missing-imports
python -m coverage run --source=src -m pytest tests -q && python -m coverage report
```

## 10. Docker

```bash
docker build -t enzyselect .
docker run --rm -p 8501:8501 enzyselect
```

Then open <http://localhost:8501>. The image ships with the committed dataset,
runs as a non-root user, and exposes a healthcheck on
`/_stcore/health`.

## 11. Screenshots

**None have been captured yet.** The list below is a placeholder — the files do
not exist in this repository. Add them to `docs/screenshots/` after running the
app locally, then link them here:

- `docs/screenshots/01-executive-summary.png` — summary tiles and disclaimer
- `docs/screenshots/02-ranked-candidates.png` — ranked table and portfolio view
- `docs/screenshots/03-deep-dive.png` — score breakdown, radar, structure viewer
- `docs/screenshots/04-scenario.png` — rank movement under changed conditions
- `docs/screenshots/05-economics.png` — illustrative cost and sensitivity

## 12. Language policy

The interface deliberately avoids the phrases *"proven enzyme"*, *"guaranteed
degradation"*, *"validated industrial performance"*, *"commercially ready"* and
*"safe for deployment"*, and prefers *candidate*, *prioritization signal*,
*illustrative score*, *requires experimental validation* and *demonstration
scenario*.

This is enforced by `tests/test_language.py`, which fails the build if the
banned vocabulary appears in any module that produces user-facing text. Wording
discipline erodes one convenient phrase at a time; a test is more reliable than
good intentions.

## 13. Suggested future improvements

**Data**
- Replace the synthetic table with a curated set of characterized PET
  hydrolases from the literature, each value carrying a citation and an
  assay-condition record. Until conditions are recorded, rates are not
  comparable across sources.
- Pull real sequences and real AlphaFold models per candidate, and compute
  actual per-residue pLDDT near the catalytic triad instead of a global mean.

**Method**
- Replace the symmetric fit with an asymmetric one whose upper arm falls off
  faster, reflecting thermal denaturation.
- Add uncertainty: every input is an estimate, so the output should be a score
  distribution and a rank *interval*, not a point estimate.
- Move from a stated-weight heuristic to a model fitted against real screening
  outcomes — and report its calibration honestly.
- Add an explicit "cost of a miss" term so the shortlist size trades off
  screening cost against the probability of discarding a viable candidate.

**Engineering**
- Persist scenarios so two rankings can be compared side by side over time.
- Add a proper structure viewer that highlights the catalytic residues.
- Batch structure prefetching with a progress indicator.

---

*EnzySelect is an educational prototype built for demonstration and portfolio
purposes. It is not a validated scientific, medical, environmental or
industrial decision-making system, and it must not be used to support real
production, procurement or investment decisions.*
