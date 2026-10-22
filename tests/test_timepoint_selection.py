"""Second-timepoint selection for the longitudinal classifier.

scripts/longitudinal/ is not a package, so both modules are loaded by path
(see test_longitudinal_followup_as_post.py). The synthetic fixtures are
under tmp_path; the real-data test reads the study data read-only and skips
when it is absent.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_LONGITUDINAL = Path(__file__).resolve().parents[1] / "scripts" / "longitudinal"


def _load(name):
    path = _LONGITUDINAL / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_selection = _load("timepoint_selection")
select_second_timepoints = _selection.select_second_timepoints
apply_second_timepoint = _selection.apply_second_timepoint


# Synthetic availability: which task orders each subject has a usable session
# for, per task and for BRIEF-A. Codes from S90 up are outside the cohort.
STUDY = {
    "task": {
        "stroop": {"S91": {1, 2}, "S93": {1, 2, 3}, "S92": {1, 3}, "S94": {2},    "S95": {1, 2}},
        "wcst":   {"S91": {1, 2}, "S93": {1, 2, 3}, "S92": {1, 3}, "S94": {1, 2}, "S95": {1, 2}},
    },
    "brief": {"S91": {1, 2}, "S93": {1, 3}, "S92": {1}, "S94": {1, 2, 3}, "S95": {1, 2}},
}


def test_study_selection():
    t2 = select_second_timepoints(STUDY["task"], STUDY["brief"]).set_index("subject_id")
    assert t2.t2_order.to_dict() == {"S91": 2, "S93": 3, "S92": 3, "S94": 2, "S95": 2}
    assert t2.loc["S93", "levels"] == "brief+stroop+wcst"
    assert t2.loc["S92", "levels"] == "stroop+wcst"
    assert t2.loc["S94", "levels"] == "brief+stroop+wcst"


def test_tie_goes_to_post():
    t2 = select_second_timepoints({"stroop": {"A": {1, 2, 3}}, "wcst": {"A": {1}}}, {"A": {1}})
    assert t2.t2_order.item() == 2


def test_baseline_only_subject_is_absent():
    t2 = select_second_timepoints({"stroop": {"A": {1}}, "wcst": {"A": {1}}}, {"A": {1}})
    assert t2.empty


def test_output_columns_and_level_count():
    t2 = select_second_timepoints(STUDY["task"], STUDY["brief"])
    assert list(t2.columns) == ["subject_id", "t2_order", "n_levels", "levels"]
    assert list(t2.subject_id) == ["S91", "S92", "S93", "S94", "S95"]
    assert t2.set_index("subject_id").n_levels.to_dict() == {
        "S91": 3, "S93": 3, "S92": 2, "S94": 3, "S95": 3,
    }


def test_apply_relabels_chosen_followup_as_post():
    df = pd.DataFrame({
        "subject_id": ["S95", "S95", "S95", "S99"],
        "task_id": ["a", "b", "c", "d"],
        "task_order": [1, 2, 3, 1],
    })
    t2 = pd.DataFrame({"subject_id": ["S95"], "t2_order": [3], "n_levels": [2],
                       "levels": ["stroop+wcst"]})

    out = apply_second_timepoint(df, t2)

    assert list(out.task_id) == ["a", "c"], "S95's order-2 row and S99 are dropped"
    assert list(out.task_order) == [1, 2]
    assert list(out.timepoint) == ["baseline", "post"]


def test_apply_drops_the_unchosen_order_when_chosen_is_post():
    # Post (order 2) is chosen; the subject also has a follow-up (order 3)
    # session. The order-3 row must be dropped outright, not relabelled to
    # order 2 alongside the real post -- that would give the subject two
    # "post" rows.
    df = pd.DataFrame({
        "subject_id": ["S99", "S99", "S99"],
        "task_id": ["a", "b", "c"],
        "task_order": [1, 2, 3],
    })
    t2 = pd.DataFrame({"subject_id": ["S99"], "t2_order": [2], "n_levels": [3],
                       "levels": ["brief+stroop+wcst"]})

    out = apply_second_timepoint(df, t2)

    assert list(out.task_id) == ["a", "b"], "S99's order-3 row is dropped, not relabelled"
    assert list(out.task_order) == [1, 2]
    assert list(out.timepoint) == ["baseline", "post"]


def _write_task_info(base_dir, task, rows):
    (base_dir / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "subj_id", "task_id", "task_order"]).to_csv(
        base_dir / "data" / f"{task}_task_info.csv", index=False
    )


def test_read_task_info_keeps_order_3(tmp_path, capsys):
    _write_task_info(tmp_path, "stroop", [
        ("f0.snirf", "S95", "s0", 1),
        ("f1.snirf", "S95", "s1", 3),
    ])
    for name in ("unified_ef_longitudinal", "longitudinal_exploratory"):
        kept = _load(name).read_task_info(tmp_path, "stroop")
        assert sorted(kept.task_order) == [1, 3]
    assert "WARNING" not in capsys.readouterr().out


def test_real_data_selection_under_main_rules(study_base_dir, reference_tree, pinned):
    _BASE_DIR, _RUN_DIR = study_base_dir, reference_tree
    if not (_BASE_DIR / "data").is_dir() or not (_RUN_DIR / "02_behav").is_dir():
        pytest.skip("study data or reference run tree incomplete")
    unified = _load("unified_ef_longitudinal")
    from scripts.session_exclusions import low_accuracy_sessions

    excluded = low_accuracy_sessions(_RUN_DIR, _BASE_DIR)
    task_orders = {}
    for task in ("stroop", "wcst"):
        info = unified.read_task_info(_BASE_DIR, task)
        dropped = info[info["task_id"].isin(excluded[task])]
        expected = sorted((subject, order) for t, subject, order
                          in pinned("low_accuracy_sessions") if t == task)
        assert sorted(zip(dropped.subject_id, dropped.task_order)) == expected
        info = info[~info["task_id"].isin(excluded[task])]
        task_orders[task] = _selection.orders_by_subject(info)

    roster = sorted(set(task_orders["stroop"]) | set(task_orders["wcst"]))
    assessments = unified.load_assessment_t_scores(
        _RUN_DIR, roster, exclude_invalid=True
    )
    t2 = select_second_timepoints(task_orders, _selection.orders_by_subject(assessments))

    assert (t2.set_index("subject_id").t2_order.to_dict()
            == pinned("longitudinal_second_timepoint_orders"))


# Questionnaire pairing at the second timepoint.
def _assessments(rows):
    return pd.DataFrame(rows, columns=["subject_id", "assessment", "metric", "value", "task_order"])


_T2_M6 = pd.DataFrame({"subject_id": ["A", "B"], "t2_order": [3, 3], "n_levels": [2, 3],
                       "levels": ["stroop+wcst", "brief+stroop+wcst"]})


def test_questionnaires_come_from_post_when_followup_has_none():
    # A: tasks at follow-up, PHQ-9/GAD-7 only at post, BRIEF-A only at post.
    df = _assessments([
        ("A", "phq9", "phq9_total_score", 12.0, 1),
        ("A", "phq9", "phq9_total_score", 0.0, 2),
        ("A", "gad7", "gad7_total_score", 12.0, 1),
        ("A", "gad7", "gad7_total_score", 0.0, 2),
        ("A", "executive_function", "executive_function_gec_t", 60.0, 1),
        ("A", "executive_function", "executive_function_gec_t", 55.0, 2),
    ])
    out = _selection.apply_second_timepoint_questionnaires(df, _T2_M6)
    q = out[out.assessment != "executive_function"]
    assert list(q.value) == [12.0, 0.0, 12.0, 0.0]
    assert list(q.task_order) == [1, 2, 1, 2]
    assert list(q.timepoint) == ["baseline", "post", "baseline", "post"]
    assert list(q.source_task_order) == [1, 2, 1, 2]
    # BRIEF-A is never substituted: the post administration is not the
    # selected follow-up, so only baseline is kept.
    ef = out[out.assessment == "executive_function"]
    assert list(ef.task_order) == [1]
    assert list(ef.source_task_order) == [1]


def test_questionnaires_at_selected_visit_are_unchanged():
    # B has PHQ-9 at both post and follow-up: the selected follow-up wins.
    df = _assessments([
        ("B", "phq9", "phq9_total_score", 7.0, 1),
        ("B", "phq9", "phq9_total_score", 5.0, 2),
        ("B", "phq9", "phq9_total_score", 6.0, 3),
        ("B", "executive_function", "executive_function_gec_t", 60.0, 1),
        ("B", "executive_function", "executive_function_gec_t", 58.0, 3),
    ])
    out = _selection.apply_second_timepoint_questionnaires(df, _T2_M6)
    expected = apply_second_timepoint(df, _T2_M6)
    pd.testing.assert_frame_equal(out.drop(columns="source_task_order"), expected)
    assert list(out.value) == [7.0, 6.0, 60.0, 58.0]
    assert list(out.source_task_order) == [1, 3, 1, 3]


def test_substitution_is_per_questionnaire():
    # PHQ-9 present at follow-up, GAD-7 only at post: only GAD-7 is substituted.
    df = _assessments([
        ("A", "phq9", "phq9_total_score", 7.0, 1),
        ("A", "phq9", "phq9_total_score", 6.0, 3),
        ("A", "gad7", "gad7_total_score", 5.0, 1),
        ("A", "gad7", "gad7_total_score", 4.0, 2),
    ])
    out = _selection.apply_second_timepoint_questionnaires(df, _T2_M6)
    assert list(out.source_task_order) == [1, 3, 1, 2]
    assert list(out.task_order) == [1, 2, 1, 2]


def test_no_post_baseline_questionnaire_leaves_baseline_only():
    df = _assessments([("A", "phq9", "phq9_total_score", 7.0, 1)])
    out = _selection.apply_second_timepoint_questionnaires(df, _T2_M6)
    assert list(out.task_order) == [1]


def test_closest_visit_ties_go_to_the_earlier_visit():
    # Selected order 3 with questionnaires at 2 and 4: equidistant, take 2.
    df = _assessments([
        ("A", "phq9", "phq9_total_score", 7.0, 1),
        ("A", "phq9", "phq9_total_score", 5.0, 2),
        ("A", "phq9", "phq9_total_score", 4.0, 4),
    ])
    out = _selection.apply_second_timepoint_questionnaires(df, _T2_M6)
    assert list(out.value) == [7.0, 5.0]
    assert list(out.source_task_order) == [1, 2]
