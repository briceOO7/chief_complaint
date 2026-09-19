"""
config.py
---------
Single source of truth for paths and constants.

Reads local_paths.cfg (gitignored) for machine-specific paths.
"""

import configparser
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── Local machine paths (from local_paths.cfg or env var) ─────────────────────

def _medevac_db_root() -> Path | None:
    env = os.environ.get("MEDEVAC_DB_DIR")
    if env:
        return Path(env)
    cfg_path = ROOT / "local_paths.cfg"
    if cfg_path.exists():
        cfg = configparser.ConfigParser()
        # Explicit encoding — see llm_cedis_second_labeller.py's
        # _load_local_paths() for why (Windows locale-encoding mojibake).
        cfg.read(cfg_path, encoding="utf-8-sig")
        try:
            return Path(cfg["paths"]["medevac_db"])
        except KeyError:
            pass
    return None


MEDEVAC_DB_ROOT = _medevac_db_root()
DATA_MODE       = os.environ.get("MEDEVAC_DATA_MODE", "medevac")

# ── Project-local directories ──────────────────────────────────────────────────

DATA_DIR        = ROOT / "data"
HAND_LABELS_DIR = DATA_DIR / "hand_labels"
RAW_DIR         = DATA_DIR / "raw"          # PHI inputs — gitignored
RESULTS_DIR     = DATA_DIR / "results"
ABBREV_DIR      = DATA_DIR / "abbreviations"
MODELS_DIR      = ROOT / "models"

for _d in [RAW_DIR, RESULTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Key input files ────────────────────────────────────────────────────────────

# Deidentified corpus — tracked in git, used for BERT training
DEID_CORPUS_FILE = HAND_LABELS_DIR / "chief_complaints_medevac_all_corpus_deid.csv"

# Gold labels (from medevac_labelled.csv row-aligned to the corpus).
# This is the most recent complete gold label file; update as new labels are added.
GOLD_LABELS_FILE = HAND_LABELS_DIR / "gold_labelled_combined_20260302_134139_n8819.csv"

TRAIN_FILE = HAND_LABELS_DIR / "train.csv"
TEST_FILE  = HAND_LABELS_DIR / "test.csv"

ABBREV_FILE = ABBREV_DIR / "medevac_abbreviations.csv"

# ── medevac_db inputs/outputs (require local_paths.cfg) ───────────────────────

def medevac_db_raw_cc() -> Path:
    """Raw chief complaints from medevac_db ETL (PHI)."""
    if MEDEVAC_DB_ROOT is None:
        raise RuntimeError(
            "medevac_db path not configured.\n"
            "Create local_paths.cfg with [paths] medevac_db = /path/to/medevac_db"
        )
    return MEDEVAC_DB_ROOT / "data" / "final" / DATA_MODE / "observations_chief_complaints_phi.csv"


def medevac_db_labelled_output() -> Path:
    """Where the final labeled chief_complaints_phi.csv is written back to medevac_db."""
    if MEDEVAC_DB_ROOT is None:
        raise RuntimeError("medevac_db path not configured — see local_paths.cfg")
    return MEDEVAC_DB_ROOT / "data" / "final" / DATA_MODE / "chief_complaints_phi.csv"


def medevac_db_gold_labels() -> Path:
    """medevac_labelled.csv in medevac_db (the row-aligned gold label file)."""
    if MEDEVAC_DB_ROOT is None:
        raise RuntimeError("medevac_db path not configured — see local_paths.cfg")
    return MEDEVAC_DB_ROOT / "data" / "final" / DATA_MODE / "medevac_labelled.csv"
