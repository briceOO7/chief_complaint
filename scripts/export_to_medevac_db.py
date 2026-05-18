"""
export_to_medevac_db.py
------------------------
Write the final labeled chief_complaints_phi.csv back to medevac_db.

Takes the best available labeled output (panel, rule-based, or BERT),
merges CEDIS codes onto the raw observations CC file from medevac_db,
and writes the result to medevac_db/data/final/medevac/chief_complaints_phi.csv.

This replaces the row-position join used by medevac_db/scripts/attach_cedis_labels.py
with a proper key-based join on (MRN, EncounterID, MedevacDate, ValueTXT).

Usage:
    python scripts/export_to_medevac_db.py --labels data/results/panel_final.csv
    python scripts/export_to_medevac_db.py --labels data/results/panel_final.csv --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import medevac_db_raw_cc, medevac_db_labelled_output

JOIN_KEYS  = ["MRN", "EncounterID", "MedevacDate"]
TEXT_COL   = "ValueTXT"
LABEL_COLS = ["cedis_code", "cedis_complaint", "cedis_category", "text_original"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True,
                        help="Labeled CSV with JOIN_KEYS + LABEL_COLS")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without saving")
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"❌  Labels file not found: {args.labels}")
        sys.exit(1)

    # Load raw CC from medevac_db
    raw_path = medevac_db_raw_cc()
    print(f"📥 Loading raw CC from medevac_db: {raw_path}")
    raw = pd.read_csv(raw_path, dtype=str, low_memory=False)
    print(f"   {len(raw):,} rows")

    # Load labels
    print(f"📥 Loading labels: {args.labels.name}")
    labels = pd.read_csv(args.labels, dtype=str, low_memory=False)
    print(f"   {len(labels):,} rows")

    # Determine join strategy
    key_cols_present = all(c in labels.columns for c in JOIN_KEYS)
    text_col_present = TEXT_COL in labels.columns

    if key_cols_present:
        print(f"🔗 Joining on {JOIN_KEYS}")
        # Drop any existing label columns from raw before merging
        label_cols_to_add = [c for c in LABEL_COLS if c in labels.columns]
        raw_clean = raw.drop(columns=[c for c in LABEL_COLS if c in raw.columns], errors="ignore")
        merged = raw_clean.merge(
            labels[JOIN_KEYS + label_cols_to_add].drop_duplicates(subset=JOIN_KEYS),
            on=JOIN_KEYS,
            how="left",
        )
    elif text_col_present:
        print(f"🔗 Joining on {TEXT_COL} (key columns not all present in labels)")
        label_cols_to_add = [c for c in LABEL_COLS if c in labels.columns]
        raw_clean = raw.drop(columns=[c for c in LABEL_COLS if c in raw.columns], errors="ignore")
        merged = raw_clean.merge(
            labels[[TEXT_COL] + label_cols_to_add].drop_duplicates(subset=[TEXT_COL]),
            on=TEXT_COL,
            how="left",
        )
    else:
        print("❌  Labels file has neither JOIN_KEYS nor ValueTXT — cannot merge")
        sys.exit(1)

    n_coded = merged["cedis_code"].notna().sum() if "cedis_code" in merged.columns else 0
    print(f"\n   {len(merged):,} output rows")
    print(f"   {n_coded:,} rows with cedis_code  ({n_coded/len(merged)*100:.1f}%)")
    print(f"   {len(merged) - n_coded:,} rows without cedis_code (unlabeled)")

    out_path = medevac_db_labelled_output()
    print(f"\n📤 Output: {out_path}")

    if args.dry_run:
        print("   [dry run — not written]")
        print(f"   Columns: {list(merged.columns)}")
    else:
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"   ✅  Written: {len(merged):,} rows × {len(merged.columns)} cols")
        print(f"\n   medevac_pipeline_project can now sync this file via medevac_db's")
        print(f"   sync_to_pipeline.py (chief_complaints_phi.csv is NOT auto-synced).")
        print(f"   Copy it manually when ready:")
        print(f"   cp '{out_path}' '<pipeline_project>/data/raw/chief_complaints_phi.csv'")


if __name__ == "__main__":
    main()
