# chief_complaint

CEDIS chief complaint labeling, evaluation, and model training for medevac data.

## What this project does

Takes raw free-text chief complaints from the medevac_db ETL pipeline and applies
standardized [CEDIS](https://caep.ca/resources/cedis/) codes using three approaches:

| Part | Script | Description |
|------|--------|-------------|
| 1 | `scripts/1_label_panel.py` | LLM panel (GPT-4o-mini + Llama + Gemini) with Claude adjudicator |
| 1b | `scripts/2_label_rule_based.py` | Rule-based ensemble (rules + fuzzy + TF-IDF) — fast, offline |
| 2 | `scripts/3_evaluate_vs_labels.py` | Compare model outputs to hand-labeled gold standard |
| 2b | `scripts/export_to_medevac_db.py` | Write final labeled file back to medevac_db |
| 3 | `scripts/4_train_bert.py` | Fine-tune BERT on labeled data |
| 3b | `scripts/5_evaluate_bert.py` | Evaluate BERT performance |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `local_paths.cfg.template` → `local_paths.cfg` and set the path to your
medevac_db project:

```ini
[paths]
medevac_db = /path/to/medevac_db
```

Copy `.env.template` → `.env` and add your API keys for the LLM panel.

## Data flow

```
medevac_db/data/final/medevac/
  observations_chief_complaints_phi.csv   ← raw CC input (PHI)
          │
          ▼  scripts/1_label_panel.py  OR  2_label_rule_based.py
  data/results/
  labeled_panel_YYYYMMDD.csv             ← model predictions
          │
          ▼  scripts/3_evaluate_vs_labels.py
  data/results/
  summary_evaluation.csv                 ← performance metrics
  disagreements_*.csv                    ← rows where models disagree
          │
          ▼  scripts/export_to_medevac_db.py --labels <best output>
  medevac_db/data/final/medevac/
  chief_complaints_phi.csv               ← final labeled output
```

## Key files

| File | Description |
|------|-------------|
| `data/hand_labels/gold_labelled_combined_20260302_134139_n8819.csv` | Gold standard labels (8,819 rows) |
| `data/hand_labels/chief_complaints_medevac_all_corpus_deid.csv` | Deidentified corpus for BERT training |
| `data/hand_labels/train.csv` / `test.csv` | Train/test split for BERT |
| `data/abbreviations/medevac_abbreviations.csv` | Alaska-specific medical abbreviations |
| `src/cc_labeler/cedis_codes.py` | Complete CEDIS V2.0 code lookup (180+ codes) |

## PHI laptop setup

This project syncs via GitHub. **No PHI ever goes through git.**
PHI data is read directly from medevac_db on the same machine.

On the PHI laptop (Windows), after `git pull`:

```powershell
# 1. One-time setup
copy local_paths.cfg.template local_paths.cfg
# Edit local_paths.cfg — set medevac_db to your Windows path:
#   medevac_db = C:/Users/brian.rice/Python Projects/medevac_db

# 2. Copy .env.template → .env and add API keys (for LLM panel only)
copy .env.template .env

# 3. Install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Scripts read PHI directly from medevac_db via `local_paths.cfg` —
nothing PHI is ever copied into this project folder or committed to git.
Results in `data/results/` contain only aggregated stats, no PHI.

## PHI policy

- `data/raw/` is gitignored — PHI files live here only on the PHI laptop
- `data/hand_labels/medevac_labelled.csv` is gitignored — PHI-adjacent
- Deidentified corpus and gold labels are tracked in git
- Never commit files with MRN, DOB, or free-text patient notes

## Relationship to other projects

- **medevac_db**: ETL source. Produces `observations_chief_complaints_phi.csv`.
  This project reads from it and writes `chief_complaints_phi.csv` back.
- **medevac_pipeline_project**: Consumer. Reads `chief_complaints_phi.csv` from
  `data/raw/` and links CEDIS codes to patient journeys via `link_cedis_to_journeys.py`.
  Copy the output manually when ready (not auto-synced).
