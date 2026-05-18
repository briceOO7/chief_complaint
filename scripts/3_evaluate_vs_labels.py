"""
3_evaluate_vs_labels.py
------------------------
Compare model predictions to hand-labeled gold standard.

Reads one or more model output CSVs (from scripts 1 and 2) and the gold
labels, then reports per-model performance: accuracy, Cohen's kappa, F1
(macro/weighted), and a confusion matrix saved to data/results/.

Gold label source: data/hand_labels/gold_labelled_combined_*.csv
  Required columns: text_original (or ValueTXT), cedis_code

Model output format (any CSV with these columns):
  text_original (or ValueTXT), cedis_code_pred (or cedis_code)

Usage:
    python scripts/3_evaluate_vs_labels.py --model data/results/panel_run_*.csv
    python scripts/3_evaluate_vs_labels.py --all   # evaluate all CSVs in results/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import GOLD_LABELS_FILE, RESULTS_DIR

try:
    from scipy.stats import cohen_kappa_score
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scipy/sklearn not installed — install requirements.txt for full metrics")


SEP = "=" * 70


def _load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    text_col = "text_original" if "text_original" in df.columns else "ValueTXT"
    if text_col not in df.columns or "cedis_code" not in df.columns:
        raise ValueError(f"Gold file missing text or cedis_code columns: {list(df.columns)}")
    return df[[text_col, "cedis_code"]].rename(columns={text_col: "text"})


def _load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    # Normalise column names
    text_col = next((c for c in ["text_original", "ValueTXT", "text"] if c in df.columns), None)
    pred_col = next((c for c in ["cedis_code_pred", "cedis_code", "predicted_cedis"] if c in df.columns), None)
    if text_col is None or pred_col is None:
        raise ValueError(f"Cannot find text/prediction columns in {path.name}: {list(df.columns)}")
    return df[[text_col, pred_col]].rename(columns={text_col: "text", pred_col: "cedis_pred"})


def evaluate(gold: pd.DataFrame, preds: pd.DataFrame, label: str) -> dict:
    merged = gold.merge(preds, on="text", how="inner")
    n_gold  = len(gold)
    n_match = len(merged)

    print(f"\n{SEP}")
    print(f"  Model: {label}")
    print(SEP)
    print(f"  Gold rows          : {n_gold:,}")
    print(f"  Matched by text    : {n_match:,}  ({n_match/n_gold*100:.1f}%)")

    if n_match == 0:
        print("  ⚠️  No rows matched — check text column alignment")
        return {}

    y_true = merged["cedis_code"].fillna("unlabeled")
    y_pred = merged["cedis_pred"].fillna("unlabeled")

    results = {"model": label, "n_matched": n_match}

    if HAS_SKLEARN:
        acc = accuracy_score(y_true, y_pred)
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = float("nan")

        print(f"\n  Accuracy           : {acc:.3f}  ({acc*100:.1f}%)")
        print(f"  Cohen's kappa      : {kappa:.3f}")
        print(f"  F1 macro           : {f1_macro:.3f}")
        print(f"  F1 weighted        : {f1_weighted:.3f}")

        # Per-class breakdown (top 20 by support)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        rows = [
            (code, v["precision"], v["recall"], v["f1-score"], int(v["support"]))
            for code, v in report.items()
            if code not in ("accuracy", "macro avg", "weighted avg")
        ]
        rows.sort(key=lambda x: -x[4])
        print(f"\n  Top CEDIS codes by support (n≥5):")
        print(f"  {'Code':<8} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
        print(f"  {'-'*36}")
        for code, p, r, f1, n in rows:
            if n >= 5:
                print(f"  {code:<8} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {n:>6,}")

        results.update({"accuracy": acc, "kappa": kappa,
                        "f1_macro": f1_macro, "f1_weighted": f1_weighted})

        # Save confusion matrix
        labels_sorted = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        cm_path = RESULTS_DIR / f"confusion_{label.replace(' ', '_')}.csv"
        pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(cm_path)
        print(f"\n  Confusion matrix saved: {cm_path.name}")
    else:
        # Manual accuracy
        acc = (y_true == y_pred).mean()
        print(f"\n  Accuracy (manual)  : {acc:.3f}")
        results["accuracy"] = acc

    # Disagreement sample — rows where prediction differs from gold
    disagreements = merged[y_true.values != y_pred.values][["text", "cedis_code", "cedis_pred"]]
    dis_path = RESULTS_DIR / f"disagreements_{label.replace(' ', '_')}.csv"
    disagreements.to_csv(dis_path, index=False)
    print(f"  Disagreements saved: {dis_path.name}  ({len(disagreements):,} rows)")

    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", nargs="+", type=Path,
                       help="One or more model output CSV files to evaluate")
    group.add_argument("--all", action="store_true",
                       help="Evaluate all CSVs in data/results/")
    parser.add_argument("--gold", type=Path, default=GOLD_LABELS_FILE,
                        help=f"Gold labels CSV (default: {GOLD_LABELS_FILE.name})")
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"❌  Gold labels not found: {args.gold}")
        print("    Expected: data/hand_labels/gold_labelled_combined_*.csv")
        sys.exit(1)

    gold = _load_gold(args.gold)
    print(f"Loaded {len(gold):,} gold labels from {args.gold.name}")

    if args.all:
        model_files = sorted(RESULTS_DIR.glob("*.csv"))
        model_files = [f for f in model_files if not f.name.startswith(("confusion_", "disagreements_", "summary_"))]
    else:
        model_files = args.model

    if not model_files:
        print("No model output files found. Run scripts 1 and/or 2 first.")
        sys.exit(0)

    all_results = []
    for path in model_files:
        try:
            preds = _load_predictions(path)
            result = evaluate(gold, preds, path.stem)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"⚠️  Skipping {path.name}: {e}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{SEP}")
        print("  SUMMARY")
        print(SEP)
        summary = pd.DataFrame(all_results)
        print(summary.to_string(index=False))
        summary_path = RESULTS_DIR / "summary_evaluation.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
