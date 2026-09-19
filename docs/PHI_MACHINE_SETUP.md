# PHI Machine Setup

This guide covers how to get the chief complaint labelling pipeline running on a machine with access to raw, non-de-identified data.

## 1. Clone the repo

```bash
git clone https://github.com/<your-org>/chief_complaint.git
cd chief_complaint
```

## 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-panel.txt
```

`requirements-panel.txt` installs only what the LLM panel/labeller scripts
need (`pandas`, `requests`, `openai`, `python-dotenv`). Use the full
`requirements.txt` only if you're also running the rule-based classifier or
BERT fine-tuning on this machine — it additionally pulls in `torch`,
`transformers`, `scikit-learn`, etc., which are large, slow to install, and
unrelated to the panel/labeller workflow.

On Windows, skip venv activation if PowerShell's execution policy blocks
`Activate.ps1` — just call the venv's interpreter directly everywhere below,
e.g. `.venv\Scripts\python.exe -m pip install -r requirements-panel.txt` and
`.venv\Scripts\python.exe scripts\llm_cedis_panel.py ...`.

## 3. Configure local paths

Copy the template and edit it for this machine:

```bash
cp local_paths.cfg.template local_paths.cfg
```

Edit `local_paths.cfg`:

```ini
[paths]
raw_cc     = data/raw/chief_complaints_phi.csv   # path to your raw CC export
cc_column  = ReasonforVisitDSC                   # column name in that file
medevac_db = /path/to/medevac_db                 # optional
```

`local_paths.cfg` is gitignored and never committed.

## 4. Place your raw data

Copy your chief complaint export into `data/raw/`. That directory is gitignored — nothing in it will ever be committed.

```
data/raw/chief_complaints_phi.csv   ← your file goes here
```

## 5. Set API keys

Create a `.env` file (gitignored):

```bash
cp .env.template .env
```

Fill in:

```
PRIMARY_API_KEY=<Stanford AI Hub Standard product key>
APIM_API_KEY=<Stanford APIM key>
```

## 6. Smoke test — verify models and raw data

Run the model smoke test to confirm API access:

```bash
python scripts/smoke_test_aihub_models.py --workers 8
```

Then do a quick end-to-end test with a small slice of real data:

```bash
# Test second labeller on first 50 rows with a fast model
python scripts/llm_cedis_second_labeller.py \
    --model claude-haiku-4-5 \
    --max-rows 50

# Test panel on first 50 rows
python scripts/llm_cedis_panel.py \
    --max-rows 50 \
    --models claude-haiku-4-5 gpt-4-1-mini
```

## 7. Full run

```bash
python scripts/llm_cedis_panel.py --all --models gpt-4-1 claude-sonnet-4-6 gpt-5-4-mini
```

---

## Data layout

```
data/
  cedis_codes.csv               ← CEDIS code list including local extensions (tracked)
  abbreviations/                ← medical abbreviation expansions (tracked)
  gold_standard_v1_deid/        ← hand-labelled, human-adjudicated gold labels from
                                   the de-identified medevac corpus (tracked, READ-ONLY)
  raw/                          ← PHI input data (gitignored)
    README.md
    chief_complaints_phi.csv    ← place your export here
  results/                      ← model output (gitignored on PHI machine)
```

> **Do not modify `data/gold_standard_v1_deid/`.**
> This is the completed v1 gold standard from the de-identified medevac corpus.
> If you produce a new gold-standard dataset from PHI data, save it as `data/gold_standard_v2_phi/` (gitignored).

## 8. Reconciling gold labels with a fresh 2+1 panel run (v2 gold standard)

`medevac_labelled.csv` (medevac_db's existing hand-adjudicated medevac gold
labels) predates the current CEDIS local-code numbering (888/889/891 —
retirement of the old 890 — see `docs/CEDIS_LABELLING_RULES.md`). To produce
a reconciled v2 gold standard without touching the original:

```bash
# 1. Run the 2+1 panel on real medevac data (via medevac_db's
#    run_all_pipelines.py pause, or directly here):
python scripts/llm_cedis_panel.py --all --cohort medevac \
    --out-dir data/gold_standard_v2_phi/panel_runs/medevac_<stamp>/

# 2. Diff the panel's output against the existing gold labels:
python scripts/reconcile_gold_v2.py \
    --panel-results data/gold_standard_v2_phi/panel_runs/medevac_<stamp>/panel_results.csv
# → data/gold_standard_v2_phi/reconciliation_<stamp>/discrepancies_for_review.csv

# 3. Open discrepancies_for_review.csv and fill in the `decision` column for
#    every row: agree_with_panel | keep_gold | disagree
#    (disagree = a genuine, considered disagreement kept for manuscript
#    reporting — not resolved to either code. Optionally fill
#    `corrected_code` if neither existing label is right.)

# 4. Produce the final reconciled file:
python scripts/apply_gold_v2_decisions.py \
    --reconciliation-dir data/gold_standard_v2_phi/reconciliation_<stamp>/
# → .../gold_labelled_v2.csv
```

This never modifies `medevac_labelled.csv` or `data/gold_standard_v1_deid/`.
`gold_labelled_v2.csv` is a candidate — promote it manually (e.g. copy it
over medevac_db's `medevac_labelled.csv` and re-run `attach_cedis_labels.py`)
once you're satisfied with it.

## Local extension codes

These codes extend standard CEDIS V2.0 (which ends at 869):

| Code | Label |
|---|---|
| 888 | Follow-up/Return Visit |
| 889 | Well visit |
| 891 | Planned telehealth |
| 999 | Unknown / unclassifiable |

See `docs/CEDIS_LABELLING_RULES.md` for the full disambiguation ruleset used in model prompts.
