"""BRIEF-A figures leave out administrations that fail validity.

The plot loader must blank administrations that fail a validity scale, not
only unadministered rows, so that a figure never shows a score every
analysis excludes. Fixture CSV under tmp_path; subject codes from S90 up are
outside the cohort's range.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "assessments"
           / "01_plot_executive_function_t-score.py")


@pytest.fixture
def plot(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("plot_brief", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_administration_is_not_plotted(plot, tmp_path):
    row = {c: 50 for c in plot.SUBSCALE_LIST}
    df = pd.DataFrame([
        {"subject": "S91", "total_score": 130, "n_items": 75, **row,
         "inconsistency_acceptable": True, "Infrequency_acceptable ": True,
         "negativity_acceptable": True},
        {"subject": "S92", "total_score": 75, "n_items": 75, **row,
         "inconsistency_acceptable": True, "Infrequency_acceptable ": False,
         "negativity_acceptable": True},
        {"subject": "S93", "total_score": 0, "n_items": 0, **row,
         "inconsistency_acceptable": True, "Infrequency_acceptable ": True,
         "negativity_acceptable": True},
    ])
    df.to_csv(tmp_path / plot.TIMEPOINT_FILE_MAP["post"], index=False)

    loaded = plot.load_timepoint_scores("post", tmp_path)

    assert loaded["subject"].tolist() == ["S91"]
