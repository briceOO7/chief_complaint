"""Protected visit codes (888 Follow-up/Return Visit, 889 Well visit,
891 Planned telehealth) survive the sparse-code filter in
scripts/train_bert_cedis.py. Synthetic rows only; no models, no PHI."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tb():
    spec = importlib.util.spec_from_file_location(
        "train_bert_cedis_under_test", ROOT / "scripts" / "train_bert_cedis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(counts: dict[str, int]) -> pd.DataFrame:
    rows = [
        {"text": f"synthetic {code} {i}", "cedis_code": code}
        for code, n in counts.items()
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_protected_codes_are_the_three_visit_codes(tb):
    assert tb.PROTECTED_CODES == {"888", "889", "891"}


def test_sparse_protected_codes_are_kept(tb):
    df = _frame({"003": 10, "888": 1, "889": 2, "891": 1})
    out = tb.drop_sparse_codes(df)
    assert set(out["cedis_code"]) == {"003", "888", "889", "891"}
    assert len(out) == len(df)


def test_other_sparse_codes_are_still_dropped(tb):
    df = _frame({"003": 10, "004": 2, "888": 1})
    out = tb.drop_sparse_codes(df)
    assert set(out["cedis_code"]) == {"003", "888"}
    assert len(out) == 11


def test_no_sparse_codes_leaves_frame_unchanged(tb):
    df = _frame({"003": 5, "004": 3})
    out = tb.drop_sparse_codes(df)
    assert len(out) == len(df)


def test_can_stratify_false_with_singleton_class(tb):
    labels = np.array([0] * 10 + [1] * 10 + [2])
    assert tb.can_stratify(labels, 0.20) is False


def test_can_stratify_true_for_normal_classes(tb):
    labels = np.array([0] * 10 + [1] * 10 + [2] * 5)
    assert tb.can_stratify(labels, 0.20) is True


def test_load_annotations_keeps_sparse_visit_codes_and_splits(tb, tmp_path):
    df = _frame({"003": 12, "004": 12, "888": 1, "889": 1, "891": 2, "005": 2})
    path = tmp_path / "gold.csv"
    df.to_csv(path, index=False)
    out = tb.load_annotations(path, "cedis_code")
    assert {"888", "889", "891"} <= set(out["cedis_code"])
    assert "005" not in set(out["cedis_code"])
    assert (out["cedis_code"] == "888").sum() == 1
    assert (out["cedis_code"] == "889").sum() == 1
    assert (out["cedis_code"] == "891").sum() == 2
