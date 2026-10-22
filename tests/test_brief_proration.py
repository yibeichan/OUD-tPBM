"""BRIEF-A missing items are prorated and counted; double responses score the lower code.

A missing (``-``) response takes the mean of the scale's answered items,
rounded to the nearest integer, so a scale with one item missing is not
summed over fewer items. A double response such as ``N S`` scores the lower
of its codes. Synthetic fixtures are under tmp_path; the real-sheet check
needs OUD_TPBM_BASE_DIR and the local pinned-values file.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "assessments"


def _load(name, tmp_path, monkeypatch, base_dir=None):
    monkeypatch.setenv("BASE_DIR", str(base_dir or tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loader(tmp_path, monkeypatch):
    return _load("00_load_assessment_data.py", tmp_path, monkeypatch)


@pytest.fixture
def scorer(tmp_path, monkeypatch):
    return _load("01_analyze_executive_function.py", tmp_path, monkeypatch)


def test_double_response_scores_the_lower_code(loader):
    assert loader.convert_response_to_numeric("N S") == 1
    assert loader.convert_response_to_numeric("n o") == 1
    assert loader.convert_response_to_numeric("O S") == 2
    with pytest.raises(ValueError, match="'N X'"):
        loader.convert_response_to_numeric("N X")


def _administration(subject, overrides):
    """75 answered items (all 'Sometimes' = 2) with some items overridden."""
    values = {i: 2.0 for i in range(1, 76)}
    values.update(overrides)
    return [{"Item_Number": i, "Question": f"q{i}", subject: v} for i, v in values.items()]


# Synthetic Working Memory items, with item 26 missing: 7 answered, sum 16.
# Subject codes from S90 up are outside the cohort's range.
SYNTHETIC_WM = {4: 2.0, 11: 3.0, 17: 2.0, 26: np.nan, 35: 3.0, 46: 2.0, 56: 2.0, 68: 2.0}


def test_missing_item_is_prorated_from_the_scale_mean(scorer):
    scores = scorer.calculate_scores({"baseline": _administration("S91", SYNTHETIC_WM)},
                                     "baseline")

    wm = scores["subscales"]["Working Memory"]
    # mean of the answered seven is 16/7 = 2.29, rounded to 2: raw 16 -> 18
    assert wm["raw_score"] == 18
    assert wm["n_prorated"] == 1
    assert wm["n_items"] == 7
    assert scores["n_items"] == 74
    assert scores["total_score"] == 2 * 75 + 2  # items 11 and 35 are 3
    assert all(d["n_prorated"] == 0 for s, d in scores["subscales"].items()
               if s != "Working Memory")


def test_half_rounds_up(scorer):
    # Shift has 6 items; four answered with mean 2.5, which rounds up to 3.
    overrides = {8: 3.0, 22: 2.0, 32: 3.0, 44: 2.0, 61: np.nan, 67: np.nan}
    scores = scorer.calculate_scores({"baseline": _administration("S92", overrides)}, "baseline")
    assert scores["subscales"]["Shift"]["raw_score"] == 10 + 3 + 3
    assert scores["subscales"]["Shift"]["n_prorated"] == 2


def test_unadministered_scale_is_not_prorated(scorer):
    overrides = {i: np.nan for i in range(1, 76)}
    scores = scorer.calculate_scores({"post": _administration("S93", overrides)}, "post")
    assert scores["n_items"] == 0
    assert all(d["n_prorated"] == 0 for d in scores["subscales"].values())


def test_prorated_counts_reach_the_scores_csv(scorer, tmp_path):
    df = scorer.compile_subject_scores({"S91": {"baseline": _administration("S91", SYNTHETIC_WM)}})
    scorer.write_timepoint_csvs(df)
    out = pd.read_csv(tmp_path / "out" / "01_assessments" / "scores"
                      / "executive_function_baseline_subscale_scores.csv")
    assert out.loc[0, "Working Memory"] == 18
    assert out.loc[0, "Working Memory_n_prorated"] == 1


def test_real_baseline_sheet(tmp_path, monkeypatch, study_base_dir, pinned):
    """Scores from the real baseline sheet match the locally pinned values."""
    loader = _load("00_load_assessment_data.py", tmp_path, monkeypatch, base_dir=study_base_dir)
    data = loader.process_assessment_data(
        "executive_function", ["executive_function_baseline.csv"])
    scorer = _load("01_analyze_executive_function.py", tmp_path, monkeypatch,
                   base_dir=study_base_dir)
    serialised = json.loads(json.dumps(loader.serialize_assessment_data(data)))
    df = scorer.compile_subject_scores(serialised).set_index("subject")

    for subject, expected in pinned("brief_baseline_scores").items():
        for column, value in expected.items():
            assert df.loc[subject, column] == value, (subject, column)
    prorated_cols = [c for c in df.columns if c.endswith("_n_prorated")]
    others = df.drop(index=pinned("brief_baseline_prorated_subjects"))[prorated_cols]
    assert (others == 0).all().all()
