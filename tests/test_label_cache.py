"""
Label-cache tests for scripts/llm_cedis_panel.py using fake Bedrock clients.

No network, no real data: every model call goes to FakeClient, which labels
synthetic texts deterministically and records exactly which texts it saw.

    python -m pytest tests
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import cedis_label_cache  # noqa: E402
import llm_cedis_panel as panel  # noqa: E402

# Synthetic complaint -> (haiku code, sonnet code); arbiter answers below.
PANEL_VOTES = {
    "chest pain":      (3, 3),
    "abdominal pain":  (251, 251),
    "fever":           (257, 257),
    "fall off roof":   (401, 402),   # dispute, arbiter sides with haiku
    "strange feeling": (254, 256),   # dispute, arbiter overrules both
}
ARBITER = {"fall off roof": 401, "strange feeling": 999}
LABELLED_COLUMNS = ["row", "text", "text_original", "cedis_code", "cedis_complaint",
                    "needs_hand_review"]


class FakeCalls:
    def __init__(self):
        self.log: list[tuple[str, list[str]]] = []

    def texts(self, model: str | None = None) -> list[str]:
        return [t for m, ts in self.log if model is None or m == model for t in ts]


def _lookup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class FakeClient:
    def __init__(self, model: str, calls: FakeCalls):
        self.model, self.calls = model, calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        user = kwargs["messages"][-1]["content"]
        if user.startswith("Adjudicate"):
            texts = re.findall(r'^\d+\. "(.*)"$', user, flags=re.M)
            results = [{"cedis_code": ARBITER[_lookup(t)], "cedis_complaint": "x",
                        "rationale": "fake arbiter"} for t in texts]
        else:
            texts = re.findall(r"^\d+\. (.*)$", user, flags=re.M)
            idx = 0 if "haiku" in self.model else 1
            results = [{"cedis_code": PANEL_VOTES[_lookup(t)][idx], "cedis_complaint": "x",
                        "confidence": 3} for t in texts]
        print(f"FAKE_API_CALL {self.model} n={len(texts)}")
        self.calls.log.append((self.model, texts))
        msg = SimpleNamespace(content=json.dumps({"results": results}))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


@pytest.fixture
def calls(monkeypatch):
    c = FakeCalls()
    monkeypatch.setattr(panel, "build_client", lambda model: FakeClient(model, c))
    monkeypatch.setattr(panel.time, "sleep", lambda s: None)
    return c


@pytest.fixture
def work(tmp_path):
    return tmp_path


def _write_raw(path: Path, texts: list[str]) -> Path:
    pd.DataFrame({"ValueTXT": texts}).to_csv(path, index=False)
    return path


def _run(monkeypatch, work: Path, texts: list[str], *extra: str, out: str = "out",
         chunk_size: int = 3) -> pd.DataFrame:
    raw = _write_raw(work / "raw.csv", texts)
    out_dir = work / out
    if out_dir.exists() and "--resume" not in extra:
        shutil.rmtree(out_dir)
    argv = ["llm_cedis_panel.py", "--all", "--cohort", "commercial", "--raw", str(raw),
            "--abbrev", "none", "--out-dir", str(out_dir),
            "--cache-dir", str(work / "cache"), "--chunk-size", str(chunk_size), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    panel.main()
    return pd.read_csv(out_dir / "commercial_labelled.csv")


def _cache_lines(work: Path) -> list[dict]:
    path = work / "cache" / "commercial.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


BASE = ["chest pain", "Chest  Pain ", "abdominal pain", "fall off roof",
        "strange feeling", "fever", "chest pain"]


def test_normalized_text_key():
    k = cedis_label_cache.text_key
    assert k("Chest  Pain ") == k("chest pain") == k("CHEST\tPAIN")
    assert k("chest pain") != k("chest pains")


def test_first_run_labels_each_distinct_text_once(monkeypatch, work, calls):
    out = _run(monkeypatch, work, BASE)
    haiku = calls.texts("claude-haiku-4-5")
    assert len(haiku) == 5
    assert sorted(_lookup(t) for t in haiku) == sorted(PANEL_VOTES)
    assert sorted(_lookup(t) for t in calls.texts("claude-opus-4-6")) == \
        ["fall off roof", "strange feeling"]

    assert list(out.columns) == LABELLED_COLUMNS
    assert out["row"].tolist() == list(range(1, 8))
    assert out["text"].tolist() == BASE
    by_text = dict(zip(out["text"].map(_lookup), out["cedis_code"]))
    assert by_text["chest pain"] == 3
    assert by_text["fall off roof"] == 401
    strange = out[out["text"] == "strange feeling"].iloc[0]
    assert bool(strange["needs_hand_review"]) is True
    assert pd.isna(strange["cedis_code"])
    assert out["needs_hand_review"].sum() == 1
    assert len(_cache_lines(work)) == 5


def test_second_run_makes_no_api_calls_and_matches(monkeypatch, work, calls, capsys):
    first = _run(monkeypatch, work, BASE)
    calls.log.clear()
    capsys.readouterr()
    second = _run(monkeypatch, work, BASE)
    assert calls.log == []
    pd.testing.assert_frame_equal(first, second)
    printed = capsys.readouterr().out
    assert "Cached rows   : 7" in printed
    assert "New rows      : 0" in printed
    hr = pd.read_csv(work / "out" / "needs_hand_review.csv")
    assert hr["text"].tolist() == ["strange feeling"]
    full = pd.read_csv(work / "out" / "panel_results.csv")
    assert full.loc[full["text"] == "strange feeling", "arbiter_code"].iloc[0] == 999


def test_shifted_rows_only_send_new_texts(monkeypatch, work, calls, capsys):
    first = _run(monkeypatch, work, BASE[:4])
    calls.log.clear()
    capsys.readouterr()
    shifted = ["fever", "strange feeling"] + BASE[:4]
    out = _run(monkeypatch, work, shifted)
    printed = capsys.readouterr().out

    assert sorted(_lookup(t) for t in calls.texts("claude-haiku-4-5")) == \
        ["fever", "strange feeling"]
    assert printed.index("Cached rows   : 4") < printed.index("FAKE_API_CALL")
    assert "New rows      : 2  (2 distinct" in printed
    assert out["row"].tolist() == list(range(1, 7))
    old = out.iloc[2:].reset_index(drop=True)
    assert old["cedis_code"].tolist() == first["cedis_code"].tolist()
    assert old["text"].tolist() == first["text"].tolist()


def test_reextract_relabels_everything_and_keeps_old_entries(monkeypatch, work, calls, capsys):
    _run(monkeypatch, work, BASE)
    calls.log.clear()
    capsys.readouterr()
    out = _run(monkeypatch, work, BASE, "--reextract")
    printed = capsys.readouterr().out
    assert "Re-extract" in printed
    assert "New rows      : 7  (5 distinct" in printed
    assert len(calls.texts("claude-haiku-4-5")) == 5
    assert len(_cache_lines(work)) == 10
    assert out["cedis_code"].notna().sum() == 6


def test_fingerprint_change_invalidates_but_keeps_old(monkeypatch, work, calls):
    _run(monkeypatch, work, BASE)
    fp_old = {e["fingerprint"] for e in _cache_lines(work)}
    calls.log.clear()
    monkeypatch.setattr(cedis_label_cache, "METHOD_VERSION", "test-bump")
    _run(monkeypatch, work, BASE)
    assert len(calls.texts("claude-haiku-4-5")) == 5
    fps = [e["fingerprint"] for e in _cache_lines(work)]
    assert len(set(fps)) == 2 and fp_old <= set(fps)
    registry = json.loads((work / "cache" / "commercial.fingerprints.json").read_text())
    assert {v["method_version"] for v in registry.values()} == {"1", "test-bump"}


def test_fingerprint_covers_models_and_prompts():
    base, _ = panel.build_method_fingerprint("001: A")
    assert panel.build_method_fingerprint("001: A")[0] == base
    assert panel.build_method_fingerprint("001: B")[0] != base
    assert panel.build_method_fingerprint(
        "001: A", panel_models=["claude-haiku-4-5", "claude-opus-4-6"])[0] != base
    assert panel.build_method_fingerprint("001: A", arbiter_model="claude-sonnet-4-5")[0] != base


def test_resume_skips_done_chunks(monkeypatch, work, calls):
    first = _run(monkeypatch, work, BASE)
    calls.log.clear()
    chunk = sorted((work / "out" / "chunks").glob("chunk_*.csv"))[-1]
    chunk.unlink()
    second = _run(monkeypatch, work, BASE, "--resume")
    assert calls.log == []
    pd.testing.assert_frame_equal(first, second)


def test_default_cache_dir_is_gitignored():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    target = cedis_label_cache.CACHE_DIR_DEFAULT / "medevac.jsonl"
    rel = target.relative_to(REPO).as_posix()
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
    assert r.returncode == 0, f"{rel} is NOT gitignored"
