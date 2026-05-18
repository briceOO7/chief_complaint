"""
2_label_rule_based.py
----------------------
Apply the rule-based ensemble classifier (rules + fuzzy + TF-IDF) to the
raw chief complaints from medevac_db.

This is the fast, offline, deterministic baseline. Use it to:
  - Get a quick first-pass labeling without API costs
  - Compare against the LLM panel (script 1) and BERT (script 4)
  - Generate predictions for any new CC data instantly

Output: data/results/rule_based_YYYYMMDD_HHMMSS.csv
  Columns: all input columns + cedis_code, cedis_complaint, cedis_category,
           cedis_confidence, cedis_method

Usage:
    python scripts/2_label_rule_based.py
    python scripts/2_label_rule_based.py --rows 100   # quick test
    python scripts/2_label_rule_based.py --input data/raw/observations_chief_complaints_phi.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import medevac_db_raw_cc, RESULTS_DIR

TEXT_CANDIDATES = ["ValueTXT", "text_original", "ObservationValueDSC", "ReasonforVisitDSC"]


def _find_text_col(df: pd.DataFrame) -> str:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No CC text column found. Tried: {TEXT_CANDIDATES}. Got: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None,
                        help="Input CSV (default: reads from medevac_db via local_paths.cfg)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit to first N rows (for testing)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: data/results/rule_based_TIMESTAMP.csv)")
    args = parser.parse_args()

    # ── Load input ─────────────────────────────────────────────────────────────
    input_path = args.input or medevac_db_raw_cc()
    print(f"📥 Loading: {input_path}")
    df = pd.read_csv(input_path, dtype=str, low_memory=False, nrows=args.rows)
    print(f"   {len(df):,} rows")

    text_col = _find_text_col(df)
    print(f"   Text column: '{text_col}'")

    # ── Load classifier ────────────────────────────────────────────────────────
    try:
        from cc_labeler.classifier import CEDISClassifier
    except ImportError:
        # Fallback: try direct import from copied files
        sys.path.insert(0, str(ROOT / "src" / "cc_labeler"))
        from classifier import CEDISClassifier

    print("🔧 Initializing rule-based classifier...")
    clf = CEDISClassifier()

    # ── Classify ───────────────────────────────────────────────────────────────
    print(f"🏷️  Classifying {len(df):,} chief complaints...")
    results = clf.classify_dataframe(df, text_column=text_col, return_top_k=1)

    # Normalise output column names
    rename = {
        "cedis_code_1":       "cedis_code",
        "cedis_complaint_1":  "cedis_complaint",
        "cedis_category_1":   "cedis_category",
        "confidence_1":       "cedis_confidence",
        "method_1":           "cedis_method",
    }
    results = results.rename(columns={k: v for k, v in rename.items() if k in results.columns})

    # ── Summary ────────────────────────────────────────────────────────────────
    n_coded = results["cedis_code"].notna().sum()
    print(f"\n✅ Labeled {n_coded:,} / {len(results):,} rows ({n_coded/len(results)*100:.1f}%)")

    if "cedis_confidence" in results.columns:
        conf = pd.to_numeric(results["cedis_confidence"], errors="coerce")
        print(f"   Confidence — median: {conf.median():.2f}  mean: {conf.mean():.2f}")

    if "cedis_code" in results.columns:
        print(f"\n   Top 10 CEDIS codes assigned:")
        for code, n in results["cedis_code"].value_counts().head(10).items():
            print(f"     {code}: {n:,}")

    # ── Save ───────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (RESULTS_DIR / f"rule_based_{ts}.csv")
    results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved: {out_path}  ({len(results):,} rows)")
    print(f"\nNext: python scripts/3_evaluate_vs_labels.py --model {out_path}")


if __name__ == "__main__":
    main()
