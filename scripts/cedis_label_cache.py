"""
Persistent per-cohort label cache for llm_cedis_panel.py.

A chief complaint that has already been through the 2+1 panel under the same
method is never sent to the LLMs again. Each entry is keyed by

    (method fingerprint, sha256 of the normalized text)

so labels follow the TEXT, not the row position — new raw batches that shift
row numbers still hit the cache.

The method fingerprint is a sha256 over:
  - METHOD_VERSION (bump by hand whenever labelling behaviour changes in a way
    the other components don't capture, e.g. vote/abstain/arbiter logic),
  - panel + arbiter model names and their provider model IDs (api_name),
  - a hash of the prompts (classification system + user template, arbiter
    system), which embed CODING_RULES and the CEDIS code list,
  - a hash of data/cedis_codes.csv (valid codes + complaint names).

Changing any of these yields a new fingerprint, so older entries simply stop
matching. They are never deleted: every entry keeps its fingerprint, and
`{cohort}.fingerprints.json` records what each fingerprint was made of.

The cache holds real chief-complaint text (PHI on the PHI machine). It lives
under data/results/, which is gitignored — see CACHE_DIR_DEFAULT.

File format: append-only JSON Lines, one entry per line, flushed + fsynced
after every chunk so an interrupted run keeps everything labelled so far.
A truncated final line (crash mid-write) is skipped on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bump by hand when the panel's labelling behaviour changes in a way the
# fingerprint can't see (vote/abstain rules, arbiter routing, parsing, ...).
METHOD_VERSION = "1"

CACHE_DIR_DEFAULT = REPO_ROOT / "data" / "results" / "label_cache"

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Unicode-normalize, casefold and collapse whitespace."""
    t = unicodedata.normalize("NFKC", str(text))
    return _WS.sub(" ", t).strip().casefold()


def text_key(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _sha(s: str | bytes) -> str:
    b = s if isinstance(s, bytes) else s.encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def method_components(
    panel_models: list[str],
    arbiter_model: str,
    model_configs: dict,
    prompt_texts: list[str],
    cedis_codes_csv: Path,
) -> dict:
    return {
        "method_version": METHOD_VERSION,
        "panel_models": [
            {"name": m, "api_name": model_configs[m].get("api_name")} for m in panel_models
        ],
        "arbiter_model": {
            "name": arbiter_model,
            "api_name": model_configs[arbiter_model].get("api_name"),
        },
        "prompt_sha256": _sha("\n\x1e\n".join(prompt_texts)),
        "cedis_codes_sha256": _sha(Path(cedis_codes_csv).read_bytes()),
    }


def fingerprint(components: dict) -> str:
    return _sha(json.dumps(components, sort_keys=True))[:16]


def _jsonable(v):
    """pandas/numpy scalars -> plain JSON values (NA -> None)."""
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


class LabelCache:
    def __init__(self, cache_dir: Path, cohort: str, fp: str, components: dict):
        self.cache_dir = Path(cache_dir)
        self.cohort = cohort
        self.fingerprint = fp
        self.components = components
        self.path = self.cache_dir / f"{cohort}.jsonl"
        self.registry_path = self.cache_dir / f"{cohort}.fingerprints.json"
        self.entries: dict[str, dict] = {}
        self.n_other_fingerprints = 0

    def load(self) -> "LabelCache":
        """Load entries for the current fingerprint (last write wins)."""
        self.entries = {}
        self.n_other_fingerprints = 0
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("fingerprint") != self.fingerprint:
                    self.n_other_fingerprints += 1
                    continue
                self.entries[e["text_key"]] = e["result"]
        return self

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def add_many(self, items: list[tuple[str, str, dict]]) -> None:
        """Append (text_key, text, result) entries and make them visible to get()."""
        if not items:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._register_fingerprint()
        stamp = datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            for key, text, result in items:
                clean = {k: _jsonable(v) for k, v in result.items()}
                fh.write(json.dumps({
                    "fingerprint": self.fingerprint,
                    "text_key": key,
                    "text": text,
                    "labelled_at": stamp,
                    "result": clean,
                }, ensure_ascii=False) + "\n")
                self.entries[key] = clean
            fh.flush()
            os.fsync(fh.fileno())

    def _register_fingerprint(self) -> None:
        reg: dict = {}
        if self.registry_path.exists():
            try:
                reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reg = {}
        if self.fingerprint in reg:
            return
        reg[self.fingerprint] = {
            **self.components,
            "first_used": datetime.now().isoformat(timespec="seconds"),
        }
        self.registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
