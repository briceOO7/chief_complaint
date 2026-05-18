"""
train_bert_cedis.py
-------------------
Fine-tune Bio_ClinicalBERT on MANUALLY ANNOTATED CEDIS-labeled chief complaints.

⚠️  IMPORTANT — valid training data sources:
    Gold standard: a CSV of chief complaints with CEDIS codes assigned by a
    qualified clinician who reviewed the text and applied the official CEDIS
    taxonomy. See data/annotations/README.md for the annotation format.

    NOT valid: output from classify_chiefcomplaints.py (the fuzzy-match
    CEDIS classifier). That classifier is exploratory only. Using its
    predictions as training labels would teach BERT to replicate a heuristic,
    not to learn true CEDIS classification. Do not use it here.

Methodology (follows dchang56/chief_complaints, JAMIA Open 2020):
  - Base model : emilyalsentzer/Bio_ClinicalBERT (MIMIC-pretrained)
  - Task       : multi-class sequence classification → CEDIS code
  - Split      : 80% train / 10% val / 10% test
  - Metrics    : Top-1, Top-3, Top-5 accuracy (same as dchang56 paper)

Input CSV format (see data/annotations/cedis_annotations_template.csv):
    text        — free-text chief complaint (ReasonforVisitDSC or equivalent)
    cedis_code  — CEDIS code assigned by a clinician (e.g. "003")
    annotator   — optional: who assigned the label
    notes       — optional: free-text notes on ambiguous cases

Output (saved to models/bert_cedis_finetuned/):
    pytorch_model/        — HuggingFace model, loadable with from_pretrained
    label_map.json        — cedis_code → integer index mapping
    label_index.csv       — human-readable label reference
    training_summary.json — accuracy metrics + training config

Usage:
    # Train using pre-split train/test CSVs (recommended):
    python scripts/train_bert_cedis.py \
        --train data/training_cc/train.csv \
        --test  data/training_cc/test.csv

    # Train on domain labels (16 classes — use when cedis_code is too sparse):
    python scripts/train_bert_cedis.py \
        --train data/training_cc/train.csv \
        --test  data/training_cc/test.csv \
        --label-col domain

    # Train on a single combined file (script does 80/10/10 split internally):
    python scripts/train_bert_cedis.py --annotations data/training_cc/gold_labelled_combined_*.csv

    # Quick sanity-check (1 epoch, 200 rows):
    python scripts/train_bert_cedis.py \
        --train data/training_cc/train.csv \
        --test  data/training_cc/test.csv \
        --epochs 1 --max-samples 200

    # Resume from a checkpoint:
    python scripts/train_bert_cedis.py \
        --train data/training_cc/train.csv \
        --test  data/training_cc/test.csv \
        --resume models/bert_cedis_finetuned/checkpoint-epoch1

Once trained, classify with:
    python scripts/classify_cc_bert.py --dataset medevac --model finetuned

Requirements:
    pip install torch transformers sentencepiece accelerate scikit-learn tqdm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASE_MODEL    = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_DIR    = PROJECT_ROOT / "models" / "bert_cedis_finetuned"
CEDIS_CSV     = PROJECT_ROOT / "data" / "cedis_codes.csv"
MAX_SEQ_LEN   = 64
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01


def _load_cedis_lookup() -> dict:
    """Return {code_str: complaint_str} from data/cedis_codes.csv."""
    if not CEDIS_CSV.exists():
        return {}
    df = pd.read_csv(CEDIS_CSV, dtype=str)
    return {
        str(row["code"]).zfill(3): row.get("complaint", "")
        for _, row in df.iterrows()
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split_csvs(
    train_csv: Path,
    test_csv: Path,
    label_col: str,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split train/test CSVs (output of split_train_test.py).

    Required columns: text, <label_col>
    """
    for p in (train_csv, test_csv):
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)

    train_df = pd.read_csv(train_csv, dtype=str)
    test_df  = pd.read_csv(test_csv,  dtype=str)

    for name, df in [("train", train_df), ("test", test_df)]:
        if "text" not in df.columns or label_col not in df.columns:
            logger.error(
                "%s CSV is missing required columns. Need 'text' and '%s'.\n"
                "Columns found: %s",
                name, label_col, list(df.columns),
            )
            sys.exit(1)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"]    = df["text"].fillna("").str.strip()
        df[label_col] = df[label_col].fillna("").str.strip()
        before = len(df)
        df = df[(df["text"] != "") & (df[label_col] != "")]
        if label_col == "cedis_code":
            df[label_col] = df[label_col].str.zfill(3)
            df = df[df[label_col] != "999"]
        if len(df) < before:
            logger.info("Dropped %d blank/unknown rows", before - len(df))
        return df.reset_index(drop=True)

    train_df = clean(train_df)
    test_df  = clean(test_df)

    if max_samples and len(train_df) > max_samples:
        logger.info("Limiting train to %d samples (--max-samples)", max_samples)
        train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return train_df, test_df


def load_annotations(
    annotation_csv: Path,
    label_col: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """
    Load a single combined gold-label CSV and return it cleaned.

    Required columns: text, <label_col>
    """
    if not annotation_csv.exists():
        logger.error("Annotation file not found: %s", annotation_csv)
        sys.exit(1)

    logger.info("Loading annotations from %s ...", annotation_csv)
    df = pd.read_csv(annotation_csv, dtype=str)

    if "text" not in df.columns or label_col not in df.columns:
        logger.error(
            "CSV is missing required columns. Need 'text' and '%s'.\nFound: %s",
            label_col, list(df.columns),
        )
        sys.exit(1)

    df["text"]    = df["text"].fillna("").str.strip()
    df[label_col] = df[label_col].fillna("").str.strip()

    before = len(df)
    df = df[(df["text"] != "") & (df[label_col] != "")]
    if label_col == "cedis_code":
        df[label_col] = df[label_col].str.zfill(3)
        df = df[df[label_col] != "999"]
    if len(df) < before:
        logger.info("Dropped %d blank/unknown rows", before - len(df))

    if label_col == "cedis_code":
        # Require ≥3 examples per code — warn and drop sparse labels
        counts = df[label_col].value_counts()
        rare = counts[counts < 3].index
        if len(rare):
            logger.warning(
                "Dropping %d rows for codes with <3 examples: %s\n"
                "Consider using --label-col domain for sparse datasets.",
                len(df[df[label_col].isin(rare)]), sorted(rare),
            )
            df = df[~df[label_col].isin(rare)]

    cedis = _load_cedis_lookup()
    logger.info("Valid examples: %d across %d '%s' classes",
                len(df), df[label_col].nunique(), label_col)
    logger.info("Label distribution:")
    for lbl, count in df[label_col].value_counts().items():
        desc = cedis.get(str(lbl).zfill(3), lbl) if label_col == "cedis_code" else lbl
        logger.info("  %-6s  %-45s  %d", lbl, desc, count)

    if max_samples and len(df) > max_samples:
        logger.info("Limiting to %d samples (--max-samples)", max_samples)
        df = df.sample(n=max_samples, random_state=42)

    return df.reset_index(drop=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(texts, labels, tokenizer):
    import torch
    from torch.utils.data import Dataset

    encodings = tokenizer(
        texts,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class CCDataset(Dataset):
        def __getitem__(self, idx):
            return {
                "input_ids":      encodings["input_ids"][idx],
                "attention_mask": encodings["attention_mask"][idx],
                "token_type_ids": encodings.get(
                    "token_type_ids",
                    torch.zeros_like(encodings["input_ids"])
                )[idx],
                "labels": torch.tensor(labels[idx], dtype=torch.long),
            }
        def __len__(self):
            return len(labels)

    return CCDataset()


# ── Evaluation ────────────────────────────────────────────────────────────────

def top_k_accuracy(logits, labels, k: int) -> float:
    import torch
    topk = torch.topk(logits, k, dim=1).indices
    return topk.eq(labels.unsqueeze(1).expand_as(topk)).any(dim=1).float().mean().item()


def evaluate(model, dataloader, device) -> dict:
    import torch
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {
        "loss":     total_loss / len(dataloader),
        "top1_acc": top_k_accuracy(logits, labels, k=1),
        "top3_acc": top_k_accuracy(logits, labels, k=min(3, logits.shape[1])),
        "top5_acc": top_k_accuracy(logits, labels, k=min(5, logits.shape[1])),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    annotation_csv: Path | None,
    epochs: int,
    max_samples: int | None,
    resume: str | None,
    output_dir: Path,
    label_col: str = "cedis_code",
    train_csv: Path | None = None,
    test_csv:  Path | None = None,
):
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        BertForSequenceClassification,
        BertTokenizer,
        get_linear_schedule_with_warmup,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    if train_csv and test_csv:
        train_df, test_df_full = load_split_csvs(train_csv, test_csv, label_col, max_samples)
        # Carve 50% of test as a held-out val set; avoid stratifying on singletons
        if len(test_df_full) >= 4:
            singleton_labels = test_df_full[label_col].value_counts()
            singleton_labels = singleton_labels[singleton_labels < 2].index
            can_stratify = (
                test_df_full[label_col].nunique() > 1
                and len(singleton_labels) == 0
            )
            X_val_df, X_te_df = train_test_split(
                test_df_full, test_size=0.50, random_state=42,
                stratify=test_df_full[label_col] if can_stratify else None,
            )
        else:
            X_val_df = test_df_full
            X_te_df  = test_df_full
        X_tr  = train_df["text"].tolist()
        X_val = X_val_df["text"].tolist()
        X_te  = X_te_df["text"].tolist()
        # Fit LabelEncoder on ALL seen labels (train + test) to avoid unseen-class errors
        le = LabelEncoder()
        le.fit(pd.concat([train_df, test_df_full])[label_col].values)
        y_tr  = le.transform(train_df[label_col].values)
        y_val = le.transform(X_val_df[label_col].values)
        y_te  = le.transform(X_te_df[label_col].values)
        all_df = pd.concat([train_df, test_df_full])
    else:
        all_df = load_annotations(annotation_csv, label_col, max_samples)
        le = LabelEncoder()
        int_labels = le.fit_transform(all_df[label_col].values)
        texts = all_df["text"].tolist()
        # 80 / 10 / 10 split
        X_tr, X_te_all, y_tr, y_te_all = train_test_split(
            texts, int_labels, test_size=0.20, random_state=42,
            stratify=int_labels if len(le.classes_) > 1 else None,
        )
        X_val, X_te, y_val, y_te = train_test_split(X_te_all, y_te_all, test_size=0.50, random_state=42)

    num_labels = len(le.classes_)
    label_map  = {lbl: int(idx) for idx, lbl in enumerate(le.classes_)}
    logger.info("Training with %d '%s' classes", num_labels, label_col)
    logger.info("Split: %d train / %d val / %d test", len(X_tr), len(X_val), len(X_te))

    model_name = resume or BASE_MODEL
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer  = BertTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing...")
    train_ds = build_dataset(X_tr,  y_tr,  tokenizer)
    val_ds   = build_dataset(X_val, y_val, tokenizer)
    test_ds  = build_dataset(X_te,  y_te,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    logger.info("Loading model: %s", model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
    )
    model.to(device)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    logger.info("=" * 60)
    logger.info("Fine-tuning Bio_ClinicalBERT on %d annotated examples", len(X_tr))
    logger.info("Epochs: %d  |  Steps/epoch: %d", epochs, len(train_loader))
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as bar:
            for batch in bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_loss    = train_loss / len(train_loader)
        logger.info(
            "Epoch %d | train_loss=%.4f | val_top1=%.3f | val_top3=%.3f | val_top5=%.3f",
            epoch, avg_loss,
            val_metrics["top1_acc"], val_metrics["top3_acc"], val_metrics["top5_acc"],
        )
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        ckpt = output_dir / f"checkpoint-epoch{epoch}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            best = output_dir / "pytorch_model"
            model.save_pretrained(best)
            tokenizer.save_pretrained(best)
            logger.info("  *** Best model saved (val_top1=%.3f) ***", best_val_acc)

    # Final test
    logger.info("Loading best model for test evaluation...")
    best_model = BertForSequenceClassification.from_pretrained(output_dir / "pytorch_model")
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device)

    logger.info("TEST | top1=%.3f | top3=%.3f | top5=%.3f",
                test_metrics["top1_acc"], test_metrics["top3_acc"], test_metrics["top5_acc"])

    # Persist label map + summary
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    cedis = _load_cedis_lookup()
    pd.DataFrame([
        {"index": idx, label_col: lbl,
         "description": cedis.get(str(lbl).zfill(3), lbl) if label_col == "cedis_code" else lbl}
        for lbl, idx in sorted(label_map.items(), key=lambda x: x[1])
    ]).to_csv(output_dir / "label_index.csv", index=False)

    summary = {
        "base_model":      BASE_MODEL,
        "label_col":       label_col,
        "annotation_file": str(annotation_csv or train_csv),
        "total_annotated": len(all_df),
        "num_labels":      num_labels,
        "train_examples":  len(X_tr),
        "val_examples":    len(X_val),
        "test_examples":   len(X_te),
        "epochs":          epochs,
        "max_seq_length":  MAX_SEQ_LEN,
        "batch_size":      BATCH_SIZE,
        "learning_rate":   LEARNING_RATE,
        "test_top1_acc":   round(test_metrics["top1_acc"], 4),
        "test_top3_acc":   round(test_metrics["top3_acc"], 4),
        "test_top5_acc":   round(test_metrics["top5_acc"], 4),
        "training_history": history,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Done. Model → %s", output_dir / "pytorch_model")
    logger.info("Test top-1: %.1f%%", test_metrics["top1_acc"] * 100)
    logger.info("Test top-3: %.1f%%", test_metrics["top3_acc"] * 100)
    logger.info("Test top-5: %.1f%%", test_metrics["top5_acc"] * 100)
    logger.info("Next step:  python scripts/classify_cc_bert.py --model finetuned")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Bio_ClinicalBERT on gold-labelled CEDIS chief complaints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input: pre-split files (preferred) or single combined file ────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--train", type=Path, dest="train_csv",
        help="Pre-split training CSV (output of split_train_test.py)",
    )
    parser.add_argument(
        "--test", type=Path, dest="test_csv",
        help="Pre-split test CSV (required when --train is given)",
    )
    input_group.add_argument(
        "--annotations", type=Path,
        help="Single combined CSV (script does 80/10/10 split internally)",
    )

    # ── Label target ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--label-col", default="domain",
        choices=["domain", "cedis_code"],
        help=(
            "Column to use as the classification target.\n"
            "  domain     — 16 broad classes (recommended for <1000 rows)\n"
            "  cedis_code — 84 specific codes (needs more data per class)\n"
            "Default: domain"
        ),
    )

    # ── Training config ───────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training data at N rows (for quick tests)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint directory to resume from")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Where to save the trained model (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    # Validate input combination
    if args.train_csv and not args.test_csv:
        parser.error("--test is required when --train is provided")
    if not args.train_csv and not args.annotations:
        # Default: use our standard split files
        args.train_csv = PROJECT_ROOT / "data" / "training_cc" / "train.csv"
        args.test_csv  = PROJECT_ROOT / "data" / "training_cc" / "test.csv"
        logger.info("No input specified — using default split files:\n  %s\n  %s",
                    args.train_csv, args.test_csv)

    try:
        import torch, transformers  # noqa: F401
    except ImportError:
        logger.error("Run: pip install torch transformers sentencepiece accelerate scikit-learn")
        sys.exit(1)

    train(
        annotation_csv=args.annotations,
        epochs=args.epochs,
        max_samples=args.max_samples,
        resume=args.resume,
        output_dir=args.output_dir,
        label_col=args.label_col,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
    )


if __name__ == "__main__":
    main()
