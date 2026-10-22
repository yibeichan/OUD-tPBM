"""Assessment summary JSONs count only administrations with a score.

n_subjects counts only subject rows with data at a timepoint, and a missing
PHQ-9 item 9 is missing, not "no suicidal ideation". Synthetic frames under
tmp_path.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "assessments"


def _load(name, tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(name.replace("0", "m", 1), _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["02_analyze_phq9", "03_analyze_gad7",
                                  "04_analyze_craving", "05_analyze_tea"])
def test_n_subjects_counts_scored_rows_only(name, tmp_path, monkeypatch):
    module = _load(name, tmp_path, monkeypatch)
    df = pd.DataFrame({
        "subject": ["S1", "S2", "S3"], "timepoint": ["post"] * 3,
        "total_score": [5.0, np.nan, 7.0], "n_items": [9, 0, 9],
        "severity": ["Mild", "Unknown", "Mild"],
        "suicidal_ideation": [False, np.nan, True],
        "item_1": [1.0, np.nan, 2.0],
    })
    summary = module.generate_summary_statistics(df)
    assert summary["post"]["n_subjects"] == 2


def test_phq9_missing_item_9_is_unknown_and_rate_uses_answered(tmp_path, monkeypatch):
    phq = _load("02_analyze_phq9", tmp_path, monkeypatch)
    df = pd.DataFrame({
        "subject": ["S1", "S2", "S3"], "timepoint": ["post"] * 3,
        "total_score": [5.0, np.nan, 7.0], "severity": ["Mild", "Unknown", "Mild"],
        "suicidal_ideation": [False, np.nan, True], "item_1": [1.0, np.nan, 2.0],
    })
    summary = phq.generate_summary_statistics(df)
    assert summary["post"]["suicidal_ideation_rate"] == pytest.approx(50.0)
    assert summary["post"]["severity_distribution"] == {"Mild": 2}
