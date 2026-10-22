"""Tests for longitudinal_exploratory.py's change scores and ROI source.

- ``compute_within_subject_changes`` differences the two timepoints of the
  same subject, task and metric, never rows of different metrics.
- ``load_roi_brain_activation`` reads the all-timepoints ROI file, so a
  second-timepoint value exists for every session that has one.

Subject codes and values are synthetic. The module is loaded by path and
calls ``load_dotenv()`` only inside ``main()``; every fixture is under
``tmp_path``.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "longitudinal" / "longitudinal_exploratory.py")


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("longitudinal_exploratory", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _long(rows):
    df = pd.DataFrame(rows, columns=["subject_id", "task", "task_order", "metric", "value"])
    df["timepoint"] = df["task_order"].map({1: "baseline", 2: "post"})
    return df


def test_changes_are_computed_within_each_metric(module):
    df = _long([
        ("S91", "wcst", 1, "categories_completed", 3),
        ("S91", "wcst", 2, "categories_completed", 8),
        ("S91", "wcst", 1, "perseverative_errors", 10),
        ("S91", "wcst", 2, "perseverative_errors", 14),
    ])
    out = module.compute_within_subject_changes(
        df, group_cols=["subject_id", "task"], value_col="value",
        direction_lookup=module.METRIC_DIRECTIONS, label_cols=["metric"])

    got = out.set_index("metric")["delta"].to_dict()
    assert got == {"categories_completed": 5, "perseverative_errors": 4}
    improved = out.set_index("metric")["improved"].to_dict()
    assert improved == {"categories_completed": True, "perseverative_errors": False}


def test_every_metric_with_a_pair_gets_a_change(module):
    """Each metric with two timepoints gets a change, adjacent rows or not."""
    metrics = ["overall_rt", "overall_accuracy", "stroop_effect_rt", "overall_rt_cv"]
    rows = [("S91", "stroop", order, m, 1.0 + order + i)
            for i, m in enumerate(metrics) for order in (1, 2)]
    out = module.compute_within_subject_changes(
        _long(rows), group_cols=["subject_id", "task"], value_col="value",
        direction_lookup=module.METRIC_DIRECTIONS, label_cols=["metric"])

    assert sorted(out["metric"]) == sorted(metrics)
    assert (out["delta"] == 1.0).all()


def test_roi_loader_prefers_the_multi_timepoint_file(module, tmp_path):
    info = pd.DataFrame({"subject_id": ["S91", "S91"], "task_id": ["pre1", "post1"],
                         "task_order": [1, 2]})
    row = dict(network_roi="Cont_PFCl", contrast="learning", activation_std=0.1)
    cross = tmp_path / "06_roi_activation" / "wcst"
    cross.mkdir(parents=True)
    pd.DataFrame([dict(subject="S91", session_id="pre1", activation=0.5, **row)]).to_csv(
        cross / "all_subjects_roi_activation_network_roi.csv", index=False)
    longi = tmp_path / "10_longitudinal" / "roi_activations_alltimepoints" / "wcst"
    longi.mkdir(parents=True)
    pd.DataFrame([dict(subject="S91", session_id="pre1", activation=0.5, **row),
                  dict(subject="S91", session_id="post1", activation=0.9, **row)]).to_csv(
        longi / "all_subjects_roi_activation_network_roi.csv", index=False)

    out = module.load_roi_brain_activation(tmp_path, info, "wcst", ["S91"],
                                           ["Cont_PFCl"], "learning")

    assert sorted(out["task_order"]) == [1, 2]


# ---------------------------------------------------------------------------
# The run's variant: second timepoint, BRIEF-A validity, GLM read rules
# ---------------------------------------------------------------------------

_VARIANT = {
    "name": "t",
    "channels": "all",
    "exclude_low_accuracy": False,
    "h2_metric": "perseverative_error_rate",
    "exclude_invalid_brief": True,
    "second_timepoint": "most_complete",
    "mask_wcst_without_regular_incorrect": False,
    "brain_direction": "efficiency",
}
_TSCORE_COLUMNS = ("Inhibit_t", "Shift_t", "Working Memory_t", "GEC_t")
_TP_FILES = {1: "baseline", 2: "post", 3: "followup"}


def _tree(tmp_path, variant):
    """S93: tasks at orders 1, 2, 3 and BRIEF-A at 1 and 3.
    S95: tasks at orders 1 and 3, BRIEF-A at 1, 2 (invalid) and 3.

    Every value is 10 * task_order, plus 1 for S95, so a change score says
    which order it was computed from.
    """
    import json

    out, base = tmp_path / "out", tmp_path / "base"
    (base / "data").mkdir(parents=True)
    out.mkdir()
    (out / "analysis_variant.json").write_text(json.dumps(variant))
    sessions = {"S93": (1, 2, 3), "S95": (1, 3)}
    for task in ("stroop", "wcst"):
        rows = [{"subj_id": s, "task_id": f"{task[0]}{s.lower()}o{o}", "task_order": o}
                for s, orders in sessions.items() for o in orders]
        pd.DataFrame(rows).to_csv(base / "data" / f"{task}_task_info.csv", index=False)
        behav = out / "02_behav" / task
        behav.mkdir(parents=True)
        for r in rows:
            v = 10.0 * r["task_order"] + (1 if r["subj_id"] == "S95" else 0)
            metrics = ({"overall_rt": v} if task == "stroop"
                       else {"perseverative_errors": v, "perseverative_error_rate": v / 100})
            (behav / f"{r['subj_id']}_{r['task_id']}_results.json").write_text(
                json.dumps({"metrics": metrics}))

    # The all-timepoints ROI file must exist and cover every session; one
    # ROI row per session is enough for these tests.
    for task, roi, contrast in (("stroop", "Cont_PFCl", "incongruent_vs_pure"),
                                ("wcst", "Cont_PFCl", "learning")):
        roi_dir = out / "10_longitudinal" / "roi_activations_alltimepoints" / task
        roi_dir.mkdir(parents=True)
        pd.DataFrame([{"subject": s, "session_id": f"{task[0]}{s.lower()}o{o}",
                       "contrast": contrast, "network_roi": roi, "network": "Cont",
                       "activation": 0.1 * o, "activation_std": 1.0, "task": task}
                      for s, orders in sessions.items() for o in orders]).to_csv(
            roi_dir / "all_subjects_roi_activation_network_roi.csv", index=False)

    brief = {"S93": {1: True, 3: True}, "S95": {1: True, 2: False, 3: True}}
    tdir = out / "01_assessments" / "t-scores"
    tdir.mkdir(parents=True)
    for order, label in _TP_FILES.items():
        rows = []
        for s, admins in brief.items():
            if order in admins:
                v = 10.0 * order + (1 if s == "S95" else 0)
                rows.append({"subject": s, "n_items": 75, **{c: v for c in _TSCORE_COLUMNS},
                             "inconsistency_acceptable": admins[order],
                             "Infrequency_acceptable ": True,
                             "negativity_acceptable": True})
        pd.DataFrame(rows).to_csv(
            tdir / f"executive_function_{label}_subscale_t-scores.csv", index=False)
    scores = out / "01_assessments" / "scores"
    scores.mkdir(parents=True)
    pd.DataFrame([{"subject": s, "timepoint": label, "total_score": 10.0 * o}
                  for s in ("S93", "S95") for o, label in _TP_FILES.items()]).to_csv(
        scores / "phq9_scores.csv", index=False)
    return out, base


def _deltas(changes, subject, metric):
    rows = changes[(changes["subject_id"] == subject) & (changes["metric"] == metric)]
    return list(rows["delta"])


def _run_main(module, monkeypatch, out, base):
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    monkeypatch.setenv("BASE_DIR", str(base))
    monkeypatch.setattr("sys.argv", ["longitudinal_exploratory.py",
                                     "--output-dir", str(out), "--base-dir", str(base)])
    module.main()
    return out / "10_longitudinal"


def test_most_complete_uses_follow_up_when_post_is_missing_or_invalid(module, monkeypatch, tmp_path):
    out, base = _tree(tmp_path, _VARIANT)
    result = _run_main(module, monkeypatch, out, base)

    behav = pd.read_csv(result / "behavior_changes.csv")
    # order 3 minus order 1: 20 for either subject; order 2 would give 10
    assert _deltas(behav, "S93", "perseverative_errors") == [20.0]
    assert _deltas(behav, "S95", "perseverative_errors") == [20.0]
    assert _deltas(behav, "S93", "overall_rt") == [20.0]
    assert set(behav["to_timepoint"]) == {"post"}

    assess = pd.read_csv(result / "assessment_changes.csv")
    # S95's post BRIEF-A fails a validity flag here, so its follow-up is used
    assert _deltas(assess, "S95", "executive_function_gec_t") == [20.0]
    assert _deltas(assess, "S93", "executive_function_gec_t") == [20.0]
    # PHQ-9 follows the same second timepoint, without the BRIEF-A validity rule
    assert _deltas(assess, "S93", "phq9_total_score") == [20.0]
    assert _deltas(assess, "S95", "phq9_total_score") == [20.0]
    assert set(assess["to_timepoint"]) == {"post"}
    assert not assess.duplicated(["subject_id", "metric"]).any()


def test_perseverative_error_rate_is_tracked(module, monkeypatch, tmp_path):
    assert "perseverative_error_rate" in module.BEHAVIOR_METRICS["wcst"]
    assert module.METRIC_DIRECTIONS[("wcst", "perseverative_error_rate")] == "lower"

    out, base = _tree(tmp_path, _VARIANT)
    result = _run_main(module, monkeypatch, out, base)
    behav = pd.read_csv(result / "behavior_changes.csv")
    assert "perseverative_error_rate" in set(behav["metric"])


def test_unchanged_value_is_not_a_failure_to_improve(module):
    """An unchanged value is neither improved nor not improved."""
    df = _long([
        ("S91", "wcst", 1, "perseverative_errors", 20),
        ("S91", "wcst", 2, "perseverative_errors", 15),
        ("S93", "wcst", 1, "perseverative_errors", 18),
        ("S93", "wcst", 2, "perseverative_errors", 18),
    ])
    out = module.compute_within_subject_changes(
        df, group_cols=["subject_id", "task"], value_col="value",
        direction_lookup=module.METRIC_DIRECTIONS, label_cols=["metric"])
    assert out.set_index("subject_id")["improved"].to_dict() == {"S91": True, "S93": None}

    summary = module.compute_consistency(out).iloc[0]
    assert summary["n_pairs"] == 1
    assert summary["n_unchanged"] == 1
    assert summary["pct_improved"] == 100.0


def test_no_always_empty_effect_size_columns(module):
    """No effect_size columns: a single pair has no per-session variance."""
    df = _long([
        ("S91", "stroop", 1, "overall_rt", 0.80),
        ("S91", "stroop", 2, "overall_rt", 0.75),
    ])
    out = module.compute_within_subject_changes(
        df, group_cols=["subject_id", "task"], value_col="value",
        direction_lookup=module.METRIC_DIRECTIONS, label_cols=["metric"])
    assert "effect_size" not in out.columns
    assert "effect_size_median" not in module.compute_consistency(out).columns


def test_consistency_counts_every_improvement_alongside_an_unchanged_pair(module):
    """An unchanged pair (improved is None) does not hide the improved pairs."""
    rows = []
    for subj, post in [("S91", 10), ("S97", 12), ("S90", 14), ("S93", 20)]:
        rows += [(subj, "wcst", 1, "perseverative_errors", 20), (subj, "wcst", 2, "perseverative_errors", post)]
    out = module.compute_within_subject_changes(
        _long(rows), group_cols=["subject_id", "task"], value_col="value",
        direction_lookup=module.METRIC_DIRECTIONS, label_cols=["metric"])
    summary = module.compute_consistency(out).iloc[0]
    assert (summary["n_pairs"], summary["n_unchanged"], summary["n_improved"]) == (3, 1, 3)
    assert summary["pct_improved"] == 100.0
