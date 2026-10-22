import json
import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis_variant import VARIANT_FILENAME
from scripts import session_exclusions as se


def _tree(tmp_path, accs, exclude=True):
    base, out = tmp_path / "base", tmp_path / "out"
    (base / "data").mkdir(parents=True)
    for task in ("stroop", "wcst"):
        rows = [(f"f{i}.snirf", s, f"{task[0]}{i:06d}", o) for i, (t, s, o, a) in enumerate(accs) if t == task]
        pd.DataFrame(rows, columns=["filename", "subj_id", "task_id", "task_order"]).to_csv(
            base / "data" / f"{task}_task_info.csv", index=False)
        (out / "02_behav" / task).mkdir(parents=True)
        for i, (t, s, o, a) in enumerate(accs):
            if t != task:
                continue
            (out / "02_behav" / task / f"{s}_{task[0]}{i:06d}_results.json").write_text(
                json.dumps({"metrics": {se.ACCURACY_METRIC[task]: a}}))
    variant = json.loads(open("config/analysis_variants/main.json").read())
    variant["exclude_low_accuracy"] = exclude
    (out / VARIANT_FILENAME).write_text(json.dumps(variant))
    return out, base


def test_at_or_below_chance_is_excluded(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S1", 1, 0.34), ("stroop", "S2", 1, 1/3),
                                 ("wcst", "S1", 1, 0.9)])
    got = se.low_accuracy_sessions(out, base)
    assert got["stroop"] == {"s000001"} and got["wcst"] == frozenset()


def test_variant_off_excludes_nothing(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S2", 1, 0.1)], exclude=False)
    assert se.excluded_sessions("stroop", out, base) == frozenset()


def test_variant_on_returns_low_accuracy_sessions(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S2", 1, 0.1)], exclude=True)
    assert se.excluded_sessions("stroop", out, base) == {"s000000"}


def test_missing_results_json_raises(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S1", 1, 0.9)])
    next((out / "02_behav" / "stroop").glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError):
        se.low_accuracy_sessions(out, base)


def test_write_exclusion_log(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S1", 1, 0.34), ("stroop", "S2", 1, 1/3),
                                 ("wcst", "S1", 1, 0.9)])
    se.write_exclusion_log(out, base)
    log_path = out / "04b_session_info" / "low_accuracy_sessions.csv"
    assert log_path.is_file()
    df = pd.read_csv(log_path)
    assert list(df.columns) == [
        "task", "subject", "session_id", "task_order", "accuracy_metric", "accuracy",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["task"] == "stroop" and row["subject"] == "S2" and row["session_id"] == "s000001"
    assert row["task_order"] == 1 and row["accuracy_metric"] == "overall_accuracy"
    assert row["accuracy"] == pytest.approx(1 / 3)


def test_write_exclusion_log_ignores_variant_flag(tmp_path):
    out, base = _tree(tmp_path, [("stroop", "S2", 1, 0.1)], exclude=False)
    se.write_exclusion_log(out, base)
    df = pd.read_csv(out / "04b_session_info" / "low_accuracy_sessions.csv")
    assert len(df) == 1 and df.iloc[0]["session_id"] == "s000000"


@pytest.mark.parametrize("bad", [float("nan"), None, "0.9"])
def test_non_finite_or_missing_accuracy_raises(tmp_path, bad):
    out, base = _tree(tmp_path, [("stroop", "S1", 1, 0.9)])
    path = next((out / "02_behav" / "stroop").glob("*.json"))
    path.write_text(json.dumps({"metrics": {se.ACCURACY_METRIC["stroop"]: bad}}))
    with pytest.raises(ValueError, match="finite"):
        se.low_accuracy_sessions(out, base)


def test_real_data_low_accuracy_sessions(study_base_dir, reference_tree, pinned):
    output_dir = reference_tree
    base_dir = study_base_dir
    if not (base_dir / "data").is_dir() or not (output_dir / "02_behav").is_dir():
        pytest.skip("study data or reference 02_behav not available")

    got = se.low_accuracy_sessions(output_dir, base_dir)

    def pairs(task):
        info = pd.read_csv(base_dir / "data" / f"{task}_task_info.csv")
        info = info[info.task_id.isin(got[task])]
        return set(zip(info.subj_id, info.task_order))

    expected = pinned("low_accuracy_sessions")
    for task in ("stroop", "wcst"):
        assert pairs(task) == {(s, o) for t, s, o in expected if t == task}
