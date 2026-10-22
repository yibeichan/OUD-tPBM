import os
from pathlib import Path
import pandas as pd
import pytest

# step2c also calls write_exclusion_log, which reads behavioral results
# (02_behav) from OUTPUT_DIR. This test's OUTPUT_DIR is a fresh scratch tree
# with no behavioral stage of its own, so the reference run tree's
# 02_behav is linked in read-only; nothing is written to it.


@pytest.fixture(scope="module")
def built(tmp_path_factory, study_base_dir, reference_tree, pinned):
    behav_source = reference_tree / "02_behav"
    if not behav_source.is_dir():
        pytest.skip("reference 02_behav results not available")
    out = tmp_path_factory.mktemp("run")
    (out / "02_behav").symlink_to(behav_source)
    import subprocess, sys
    env = dict(os.environ, OUTPUT_DIR=str(out), BASE_DIR=str(study_base_dir))
    subprocess.run([sys.executable, "scripts/brain/step2c_session_info.py"],
                   check=True, env=env, cwd=Path(__file__).resolve().parents[1])
    return out


def test_short_channel_counts_match_the_pinned_values(built, pinned):
    counts = pinned("short_channels_per_subject")
    for task in ("stroop", "wcst"):
        df = pd.read_csv(built / "04b_session_info" / f"short_channels_{task}.csv")
        per = df[df.is_short].groupby("subject").channel.nunique()
        expected = per.index.map(lambda s: counts.get(s, counts["default"]))
        assert (per.values == expected.values).all(), per
        assert (df.loc[df.is_short, "distance_mm"] < 10).all()
        assert (df.loc[~df.is_short, "distance_mm"] >= 10).all()


def test_regular_incorrect_counts_match_the_pinned_values(built, pinned):
    df = pd.read_csv(built / "04b_session_info" / "wcst_condition_counts.csv")
    zero = df[df.regular_incorrect == 0][["subject", "task_order"]]
    expected_zero = sorted(tuple(x) for x in pinned("wcst_sessions_without_regular_incorrect"))
    assert sorted(map(tuple, zero.values)) == expected_zero
    base = df[df.task_order == 1]
    assert len(base) == 12 and (base.regular_incorrect >= 1).all()
    assert ((base.regular_incorrect == 1).sum()
            == pinned("wcst_baseline_sessions_with_one_regular_incorrect"))
    ri = df.set_index(["subject", "task_order"]).regular_incorrect
    for subject, task_order, count in pinned("wcst_regular_incorrect_counts"):
        assert ri[(subject, task_order)] == count, (subject, task_order)


def test_readers(built, monkeypatch, pinned):
    monkeypatch.setenv("OUTPUT_DIR", str(built))
    import importlib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "brain"))
    si = importlib.import_module("session_info")
    df = pd.read_csv(built / "04b_session_info" / "wcst_condition_counts.csv")
    subject, task_order = pinned("wcst_sessions_without_regular_incorrect")[0]
    lacking = df[(df.subject == subject) & (df.task_order == task_order)].session_id.item()
    assert lacking in si.sessions_without_regular_incorrect()
    assert si.session_lacks_regular_incorrect(lacking) is True
    # Every baseline session has a regular_incorrect trial (checked above).
    baseline = df[df.task_order == 1].session_id.iloc[0]
    assert si.session_lacks_regular_incorrect(baseline) is False
    with pytest.raises(KeyError):
        si.session_lacks_regular_incorrect("nosuchsession")
    st = pd.read_csv(built / "04b_session_info" / "short_channels_stroop.csv")
    sess = st.session_id.iloc[0]
    counts = pinned("short_channels_per_subject")
    assert len(si.short_channels("stroop", sess)) in set(counts.values())
    with pytest.raises(KeyError):
        si.short_channels("stroop", "nosuchsession")


def test_exclusion_log_is_written_gate_independent(built, pinned):
    df = pd.read_csv(built / "04b_session_info" / "low_accuracy_sessions.csv")
    assert list(df.columns) == [
        "task", "subject", "session_id", "task_order", "accuracy_metric", "accuracy",
    ]
    rows = df[["task", "subject", "task_order"]].apply(tuple, axis=1).tolist()
    for task, subject, task_order in pinned("low_accuracy_sessions"):
        assert (task, subject, task_order) in rows
