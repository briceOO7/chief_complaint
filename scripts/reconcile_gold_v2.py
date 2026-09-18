"""
reconcile_gold_v2.py
---------------------
Compare a fresh 2+1 LLM panel run on real medevac chief-complaint data
against the EXISTING hand-labelled gold standard (medevac_db's
medevac_labelled.csv) and produce a discrepancy report for hand review.

Background
----------
medevac_labelled.csv (medevac_db/data/final/medevac/medevac_labelled.csv) was
hand-adjudicated BEFORE the current CEDIS local-code numbering was finalized
(888/889/891, retirement of 890 — see docs/CEDIS_LABELLING_RULES.md). This
script is step 2 of producing a reconciled v2 gold standard:

  1. Run the panel on real medevac data (medevac_db's
     run_llm_cedis_panel.py does this for you, or run it directly here):
         python scripts/llm_cedis_panel.py --all --cohort medevac \\
             --out-dir data/gold_standard_v2_phi/panel_runs/medevac_<stamp>/
  2. Run THIS script to diff the panel's output against medevac_labelled.csv:
         python scripts/reconcile_gold_v2.py \\
             --panel-results data/gold_standard_v2_phi/panel_runs/medevac_<stamp>/panel_results.csv
     → data/gold_standard_v2_phi/reconciliation_<stamp>/discrepancies_for_review.csv
  3. Open discrepancies_for_review.csv (Excel/Sheets) and fill in the
     `decision` column for every row (see values below).
  4. Run apply_gold_v2_decisions.py to produce the final gold_labelled_v2.csv.

`decision` column values (fill in by hand, one per discrepant row):
    agree_with_panel  — the panel's code is right; gold should update to match
    keep_gold         — the original hand-adjudicated label is right
    disagree          — genuine, considered disagreement between the human
                         labeller and the panel — recorded as-is for
                         manuscript reporting, not resolved to either code
    (leave blank)     — not yet reviewed

Optionally fill `corrected_code` if neither the gold nor panel code is
actually right — that overrides both and takes priority over `decision`.

This script never modifies medevac_labelled.csv or anything under
data/gold_standard_v1_deid/. All output goes to data/gold_standard_v2_phi/
(gitignored — see docs/PHI_MACHINE_SETUP.md).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import medevac_db_gold_labels  # noqa: E402

# Columns from panel_results.csv that are never a per-model vote column.
_KNOWN_PANEL_COLS = {
    "row", "text", "text_original", "unanimous", "n_confident", "n_distinct",
    "majority_code", "arbiter_code", "arbiter_model", "arbiter_rationale",
    "needs_hand_review", "final_code", "final_complaint",
}


def _panel_vote_cols(panel: pd.DataFrame) -> list[str]:
    """Raw panel-model vote columns (e.g. gpt-4-1-mini, claude-haiku-4-5) —
    everything that isn't a known fixed column or a `_conf` sidecar."""
    return [
        c for c in panel.columns
        if c not in _KNOWN_PANEL_COLS and not c.endswith("_conf")
    ]


def reconcile(panel_path: Path, gold_path: Path, out_dir: Path) -> None:
    print(f"📥 Panel results : {panel_path}")
    panel = pd.read_csv(panel_path, dtype=str, low_memory=False)
    print(f"   {len(panel):,} rows")

    print(f"📥 Existing gold : {gold_path}")
    gold = pd.read_csv(gold_path, dtype=str, low_memory=False)
    print(f"   {len(gold):,} rows")

    for df, name in [(panel, "panel"), (gold, "gold")]:
        if "row" not in df.columns:
            print(f"❌  {name} file has no 'row' column — cannot align.")
            sys.exit(1)

    panel = panel.copy()
    gold  = gold.copy()
    panel["row"] = panel["row"].astype(int)
    gold["row"]  = gold["row"].astype(int)

    n_panel, n_gold = len(panel), len(gold)
    if n_panel != n_gold:
        print(f"⚠️  Row count mismatch: panel={n_panel:,} gold={n_gold:,} — "
              f"proceeding with an inner join on 'row', but investigate why "
              f"these don't match (stale export? different corpus snapshot?).")

    vote_cols = _panel_vote_cols(panel)
    print(f"   Panel model vote columns detected: {vote_cols}")

    panel_keep = ["row", "text", "text_original", "final_code", "final_complaint",
                  "needs_hand_review", "arbiter_code", "arbiter_rationale"] + vote_cols
    panel_keep = [c for c in panel_keep if c in panel.columns]

    merged = gold[["row", "cedis_code", "cedis_complaint"]].rename(
        columns={"cedis_code": "gold_code", "cedis_complaint": "gold_complaint"}
    ).merge(
        panel[panel_keep].rename(
            columns={"final_code": "panel_code", "final_complaint": "panel_complaint"}
        ),
        on="row", how="inner",
    )
    print(f"   Aligned rows (inner join on 'row'): {len(merged):,}")

    merged["gold_code_i"]  = pd.to_numeric(merged["gold_code"], errors="coerce")
    merged["panel_code_i"] = pd.to_numeric(merged["panel_code"], errors="coerce")
    needs_review_bool = merged["needs_hand_review"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    # A row is "discrepant" if the codes disagree, OR the panel already
    # flagged it as needing hand review (arbiter overrode both panel
    # models) — either way a human needs to look at it, even if by
    # coincidence gold_code happens to equal panel_code's NaN (it can't).
    code_mismatch = merged["gold_code_i"] != merged["panel_code_i"]
    # pandas treats NaN != NaN as True; that's fine here (blank panel code
    # from a hand-review row is correctly always a mismatch).
    merged["discrepancy"] = code_mismatch | needs_review_bool

    n_total = len(merged)
    n_disc  = int(merged["discrepancy"].sum())
    n_match = n_total - n_disc
    print(f"\n{'='*60}")
    print(f"Compared {n_total:,} rows")
    print(f"  Agreement  : {n_match:,} ({100*n_match/n_total:.1f}%)")
    print(f"  Discrepant : {n_disc:,} ({100*n_disc/n_total:.1f}%)")
    print(f"    of which needs_hand_review (arbiter overrode both panel models): "
          f"{int(needs_review_bool.sum()):,}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows_path = out_dir / "all_rows_merged.csv"
    merged.drop(columns=["gold_code_i", "panel_code_i"]).to_csv(all_rows_path, index=False)
    print(f"\nFull merged comparison → {all_rows_path}")

    disc = merged[merged["discrepancy"]].drop(columns=["gold_code_i", "panel_code_i"]).copy()
    disc["decision"]       = ""   # fill in: agree_with_panel | keep_gold | disagree
    disc["corrected_code"] = ""   # optional override — highest priority
    disc["notes"]          = ""
    disc_path = out_dir / "discrepancies_for_review.csv"
    disc.to_csv(disc_path, index=False)
    print(f"Discrepancies for hand review → {disc_path}")

    meta = {
        "run_timestamp":  datetime.now().isoformat(timespec="seconds"),
        "panel_results":  str(panel_path),
        "gold_labels":    str(gold_path),
        "n_panel_rows":   n_panel,
        "n_gold_rows":    n_gold,
        "n_compared":     n_total,
        "n_agreement":    n_match,
        "n_discrepant":   n_disc,
        "n_needs_hand_review_flagged": int(needs_review_bool.sum()),
        "pct_agreement":  round(100 * n_match / n_total, 2),
    }
    meta_path = out_dir / "reconciliation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata → {meta_path}")

    print(f"\n{'='*60}")
    print("NEXT STEP:")
    print(f"  1. Open {disc_path}")
    print(f"  2. Fill in the 'decision' column for every row "
          f"(agree_with_panel / keep_gold / disagree)")
    print(f"  3. Run: python scripts/apply_gold_v2_decisions.py "
          f"--reconciliation-dir {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Diff a 2+1 panel run against existing medevac gold labels."
    )
    parser.add_argument("--panel-results", type=Path, required=True,
                        help="Path to panel_results.csv from llm_cedis_panel.py "
                             "(medevac cohort run)")
    parser.add_argument("--gold-labels", type=Path, default=None,
                        help="Path to medevac_labelled.csv (default: resolved "
                             "via local_paths.cfg's medevac_db path)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: "
                             "data/gold_standard_v2_phi/reconciliation_<timestamp>/)")
    args = parser.parse_args()

    if not args.panel_results.exists():
        print(f"❌  --panel-results not found: {args.panel_results}")
        sys.exit(1)

    gold_path = args.gold_labels
    if gold_path is None:
        try:
            gold_path = medevac_db_gold_labels()
        except RuntimeError as e:
            print(f"❌  {e}")
            print("    Pass --gold-labels explicitly instead.")
            sys.exit(1)
    if not gold_path.exists():
        print(f"❌  Gold labels file not found: {gold_path}")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "data" / "gold_standard_v2_phi" /
                                f"reconciliation_{stamp}")

    reconcile(args.panel_results, gold_path, out_dir)


if __name__ == "__main__":
    main()
