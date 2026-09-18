"""
apply_gold_v2_decisions.py
----------------------------
Ingest a hand-reviewed discrepancies_for_review.csv (produced by
reconcile_gold_v2.py, with the `decision` column filled in) and produce the
final v2 gold-standard label file: gold_labelled_v2.csv.

Per-row resolution (checked in this order):
    corrected_code is set          → use corrected_code               (source=corrected)
    row was not discrepant         → use gold_code, unchanged          (source=agreement)
    decision == agree_with_panel   → use panel_code                   (source=agree_with_panel)
    decision == keep_gold          → use gold_code                    (source=keep_gold)
    decision == disagree           → use gold_code, but set
                                      manuscript_disagreement=True     (source=disagree)
    decision blank/unrecognized    → ERROR listing unreviewed rows,
                                      unless --allow-unreviewed is passed
                                      (in which case cedis_code is left
                                      blank for those rows, source=unreviewed)

Never modifies medevac_labelled.csv, gold_standard_v1_deid/, or anything
outside data/gold_standard_v2_phi/.

Output columns: row, text, text_original, cedis_code, cedis_complaint,
    source, gold_code_v1, panel_code, manuscript_disagreement, notes

Usage:
    python scripts/apply_gold_v2_decisions.py \
        --reconciliation-dir data/gold_standard_v2_phi/reconciliation_20260918_.../
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VALID_DECISIONS = {"agree_with_panel", "keep_gold", "disagree"}


def apply_decisions(all_rows_path: Path, disc_path: Path, out_path: Path,
                    allow_unreviewed: bool) -> bool:
    print(f"📥 All rows        : {all_rows_path}")
    all_rows = pd.read_csv(all_rows_path, dtype=str, low_memory=False)
    print(f"   {len(all_rows):,} rows")

    print(f"📥 Reviewed discrepancies : {disc_path}")
    disc = pd.read_csv(disc_path, dtype=str, low_memory=False).fillna("")
    print(f"   {len(disc):,} rows")

    disc["decision"]       = disc["decision"].str.strip().str.lower()
    disc["corrected_code"] = disc["corrected_code"].str.strip()

    bad_decisions = disc[
        (disc["decision"] != "") & (~disc["decision"].isin(VALID_DECISIONS))
    ]
    if len(bad_decisions):
        print(f"❌  Unrecognized 'decision' values found (must be one of "
              f"{sorted(VALID_DECISIONS)} or blank):")
        print(bad_decisions[["row", "decision"]].to_string(index=False))
        return False

    unreviewed = disc[(disc["decision"] == "") & (disc["corrected_code"] == "")]
    if len(unreviewed) and not allow_unreviewed:
        print(f"\n❌  {len(unreviewed):,} discrepant row(s) have not been reviewed yet "
              f"(blank 'decision' and 'corrected_code'):")
        print(unreviewed["row"].tolist())
        print("\n   Fill in 'decision' for every row, or re-run with "
              "--allow-unreviewed to leave them blank in the output.")
        return False
    elif len(unreviewed):
        print(f"⚠️  Proceeding with {len(unreviewed):,} unreviewed row(s) left blank "
              f"(--allow-unreviewed).")

    disc_by_row = disc.set_index(disc["row"].astype(int))

    all_rows = all_rows.copy()
    all_rows["row"] = all_rows["row"].astype(int)
    all_rows["discrepancy"] = all_rows["discrepancy"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    out_rows = []
    n_by_source = {}
    for _, r in all_rows.iterrows():
        row_id = int(r["row"])
        gold_code  = r.get("gold_code", "")
        panel_code = r.get("panel_code", "")
        manuscript_disagreement = False
        notes = ""

        if not r["discrepancy"]:
            code, source = gold_code, "agreement"
        else:
            d = disc_by_row.loc[row_id] if row_id in disc_by_row.index else None
            corrected = str(d["corrected_code"]).strip() if d is not None else ""
            decision  = str(d["decision"]).strip() if d is not None else ""
            notes     = str(d["notes"]).strip() if d is not None and "notes" in d else ""

            if corrected:
                code, source = corrected, "corrected"
            elif decision == "agree_with_panel":
                code, source = panel_code, "agree_with_panel"
            elif decision == "keep_gold":
                code, source = gold_code, "keep_gold"
            elif decision == "disagree":
                code, source = gold_code, "disagree"
                manuscript_disagreement = True
            else:
                code, source = "", "unreviewed"

        n_by_source[source] = n_by_source.get(source, 0) + 1
        out_rows.append({
            "row":                     row_id,
            "text":                    r.get("text", ""),
            "text_original":           r.get("text_original", ""),
            "cedis_code":              code,
            "source":                  source,
            "gold_code_v1":            gold_code,
            "panel_code":              panel_code,
            "manuscript_disagreement": manuscript_disagreement,
            "notes":                   notes,
        })

    out_df = pd.DataFrame(out_rows).sort_values("row")

    # Map final cedis_code → canonical complaint string
    cedis_df = pd.read_csv(ROOT / "data" / "cedis_codes.csv")
    code_map = cedis_df.set_index("code")["complaint"].to_dict()
    out_df["cedis_code_i"] = pd.to_numeric(out_df["cedis_code"], errors="coerce")
    out_df["cedis_complaint"] = out_df["cedis_code_i"].apply(
        lambda x: code_map.get(int(x), "") if pd.notna(x) else ""
    )
    out_df = out_df.drop(columns=["cedis_code_i"])
    out_df = out_df[["row", "text", "text_original", "cedis_code", "cedis_complaint",
                      "source", "gold_code_v1", "panel_code",
                      "manuscript_disagreement", "notes"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Wrote {len(out_df):,} rows → {out_path}")
    print("\nBreakdown by source:")
    for src, n in sorted(n_by_source.items()):
        print(f"  {src:<18} {n:>6,}")
    n_manuscript = sum(1 for row in out_rows if row["manuscript_disagreement"])
    print(f"\nFlagged for manuscript reporting (manuscript_disagreement=True): "
          f"{n_manuscript:,}")
    print("\nThis file is a candidate v2 gold standard. It does NOT overwrite "
          "medevac_labelled.csv or data/gold_standard_v1_deid/ — promote it "
          "manually once you're satisfied with it.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply hand-reviewed decisions to produce gold_labelled_v2.csv"
    )
    parser.add_argument("--reconciliation-dir", type=Path, required=True,
                        help="Dir produced by reconcile_gold_v2.py "
                             "(contains all_rows_merged.csv + "
                             "discrepancies_for_review.csv)")
    parser.add_argument("--all-rows", type=Path, default=None,
                        help="Override path to all_rows_merged.csv")
    parser.add_argument("--discrepancies", type=Path, default=None,
                        help="Override path to discrepancies_for_review.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: "
                             "<reconciliation-dir>/gold_labelled_v2.csv)")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="Proceed even if some discrepancies haven't been "
                             "reviewed yet (left blank in output)")
    args = parser.parse_args()

    all_rows_path = args.all_rows or (args.reconciliation_dir / "all_rows_merged.csv")
    disc_path     = args.discrepancies or (args.reconciliation_dir / "discrepancies_for_review.csv")
    out_path      = args.out or (args.reconciliation_dir / "gold_labelled_v2.csv")

    for p in [all_rows_path, disc_path]:
        if not p.exists():
            print(f"❌  Not found: {p}")
            sys.exit(1)

    ok = apply_decisions(all_rows_path, disc_path, out_path, args.allow_unreviewed)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
