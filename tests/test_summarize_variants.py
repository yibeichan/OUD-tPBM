"""summarize_variants.py compares headline numbers across run trees.

All fixtures are tmp_path trees; no real run tree is touched.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.verification.summarize_variants import build_report

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verification" / "summarize_variants.py"

MAIN_VARIANT = {
    "name": "main", "channels": "long", "exclude_low_accuracy": True,
    "h2_metric": "perseverative_error_rate", "exclude_invalid_brief": True,
    "second_timepoint": "most_complete",
    "mask_wcst_without_regular_incorrect": True, "brain_direction": "efficiency",
}


def _channel_csv(rows):
    df = pd.DataFrame(rows, columns=["channel", "contrast", "significant_fdr"])
    return df


def _write_tree(root: Path, variant: dict, *, channel_survivor: bool = False):
    root.mkdir(parents=True)
    (root / "analysis_variant.json").write_text(json.dumps(variant))

    for task, contrasts in (("stroop", ["stroop_effect"]), ("wcst", ["learning"])):
        d = root / "05_channel_group_analysis" / task
        d.mkdir(parents=True)
        rows = [["S1_D1", c, False] for c in contrasts]
        if channel_survivor and task == "stroop":
            rows.append(["S9_D9", "stroop_effect", True])
        _channel_csv(rows).to_csv(d / f"{task}_all_contrasts_channel_statistics.csv", index=False)

        d = root / "07_roi_analysis" / task
        d.mkdir(parents=True)
        roi_rows = pd.DataFrame(
            [["Cont_PFCl", c, False] for c in contrasts],
            columns=["network_roi", "contrast", "significant_fdr"],
        )
        roi_rows.to_csv(d / "group_roi_statistics_network_roi.csv", index=False)

    roi_summary = pd.DataFrame([
        {"Hypothesis": "H1", "Test": "H1_self_brain", "r": 0.1, "CI_lower": -0.5,
         "CI_upper": 0.5, "p": 0.5, "p_fdr": 0.5, "n": 10, "Significant": "No"},
        {"Hypothesis": "H1", "Test": "H1_brain_behav", "r": -0.2, "CI_lower": -0.5,
         "CI_upper": 0.5, "p": 0.5, "p_fdr": 0.5, "n": 10, "Significant": "No"},
        {"Hypothesis": "H1", "Test": "H1_self_behav", "r": 0.3, "CI_lower": -0.5,
         "CI_upper": 0.5, "p": 0.5, "p_fdr": float("nan"), "n": 10, "Significant": "No"},
    ])
    d = root / "09_hypothesis_testing" / "unified_ef_model_roi"
    d.mkdir(parents=True)
    roi_summary.to_csv(d / "results_summary.csv", index=False)

    channel_summary = pd.DataFrame([
        {"Hypothesis": "H4", "Test": "H4_S9_D9_self_brain", "r": -0.5, "CI_lower": -0.9,
         "CI_upper": -0.1, "p": 0.01, "p_fdr": 0.02, "n": 10, "Significant": "Yes"},
        {"Hypothesis": "H4", "Test": "H4_S1_D2_self_brain", "r": 0.1, "CI_lower": -0.5,
         "CI_upper": 0.5, "p": 0.5, "p_fdr": 0.5, "n": 10, "Significant": "No"},
    ])
    d = root / "09_hypothesis_testing" / "unified_ef_model_channel"
    d.mkdir(parents=True)
    channel_summary.to_csv(d / "results_summary.csv", index=False)

    d = root / "10_longitudinal" / "unified_ef"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"subject_id": "S91", "t2_order": 2, "n_levels": 3, "levels": "brief+stroop+wcst"},
        {"subject_id": "S95", "t2_order": 3, "n_levels": 2, "levels": "brief+wcst"},
    ]).to_csv(d / "second_timepoint_selection.csv", index=False)

    pd.DataFrame([
        {"subject_id": "S91", "hypothesis": "H2", "convergence_class": "convergent_improvement", "n_domains": 3},
        {"subject_id": "S91", "hypothesis": "H4", "convergence_class": "partial_convergence", "n_domains": 3},
        {"subject_id": "S95", "hypothesis": "H2", "convergence_class": "convergent_improvement", "n_domains": 2},
        {"subject_id": "S95", "hypothesis": "H4", "convergence_class": "mixed", "n_domains": 2},
    ]).to_csv(d / "hypothesis_convergence.csv", index=False)

    d = root / "04b_session_info"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"task": "stroop", "subject": "S97", "session_id": "abc1234", "task_order": 1,
         "accuracy_metric": "overall_accuracy", "accuracy": 0.2},
    ]).to_csv(d / "low_accuracy_sessions.csv", index=False)

    d = root / "01_assessments" / "t-scores"
    d.mkdir(parents=True)
    baseline = pd.DataFrame([
        {"subject": "S91", "n_items": 75, "inconsistency_acceptable": True,
         "Infrequency_acceptable ": True, "negativity_acceptable": True},
        {"subject": "S95", "n_items": 75, "inconsistency_acceptable": False,
         "Infrequency_acceptable ": True, "negativity_acceptable": True},
    ])
    baseline.to_csv(d / "executive_function_baseline_subscale_t-scores.csv", index=False)
    empty = pd.DataFrame([
        {"subject": "S91", "n_items": 0, "inconsistency_acceptable": True,
         "Infrequency_acceptable ": True, "negativity_acceptable": True},
    ])
    empty.to_csv(d / "executive_function_post_subscale_t-scores.csv", index=False)
    empty.to_csv(d / "executive_function_followup_subscale_t-scores.csv", index=False)


@pytest.fixture
def two_trees(tmp_path):
    main_root = tmp_path / "run_main"
    _write_tree(main_root, MAIN_VARIANT)

    supp_variant = dict(MAIN_VARIANT, name="supp_low_accuracy", exclude_low_accuracy=False)
    supp_root = tmp_path / "run_supp"
    _write_tree(supp_root, supp_variant, channel_survivor=True)

    return {"main": main_root, "supp": supp_root}


def test_decisions_table_labels_the_one_differing_key(two_trees):
    report = build_report(two_trees)
    assert "## Analysis decisions" in report
    assert "supp_low_accuracy (exclude_low_accuracy=False)" in report
    # only the changed decision appears in the header, not every field
    assert "channels=" not in report.split("## Analysis decisions")[0]


def test_channel_survivor_counts_appear_per_task_and_contrast(two_trees):
    report = build_report(two_trees)
    section = report.split("## Channel-level group statistics")[1].split("##")[0]
    assert "stroop" in section and "stroop_effect" in section
    assert "wcst" in section and "learning" in section
    # main has 0 survivors of 1, supp has 1 survivor of 2 (extra fixture row)
    assert "0/1" in section
    assert "1/2" in section


def test_roi_survivor_counts_appear(two_trees):
    report = build_report(two_trees)
    section = report.split("## ROI-level group statistics")[1].split("##")[0]
    assert "Cont_PFCl" not in section  # counts only, not ROI names
    assert "0/1" in section


def test_confirmatory_family_excludes_tests_without_p_fdr(two_trees):
    report = build_report(two_trees)
    section = report.split("## Confirmatory ROI hypothesis tests")[1].split("##")[0]
    assert "H1_self_brain" in section
    assert "H1_brain_behav" in section
    assert "H1_self_behav" not in section  # p_fdr is NaN for this test


def test_exploratory_channel_family_reports_size_and_survivor(two_trees):
    report = build_report(two_trees)
    section = report.split("## Exploratory channel-level hypothesis tests")[1].split("## Longitudinal")[0]
    assert "1/2" in section
    assert "H4_S9_D9_self_brain" in section


def test_second_timepoint_and_convergence_grid_appear(two_trees):
    report = build_report(two_trees)
    section = report.split("## Longitudinal")[1]
    assert "t2_order" in section
    assert "S91" in section and "S95" in section
    assert "H2" in section and "H4" in section
    assert "convergent_improvement" in section


def test_headline_counts_any_pathway_and_all_three_levels(two_trees):
    report = build_report(two_trees)
    section = report.split("### Headline counts")[1]
    # S91 (n_domains=3) and S95 (n_domains=2) both have a convergent_improvement row
    assert "participants convergent on >=1 pathway" in section
    assert "2 (S91, S95)" in section
    # only S91's convergent_improvement row has n_domains == 3
    assert "participants convergent at all three levels" in section
    assert "1 (S91)" in section


def test_excluded_sessions_and_invalid_brief_listed(two_trees):
    report = build_report(two_trees)
    assert "S97" in report.split("## Excluded sessions")[1].split("##")[0]
    invalid_section = report.split("## Invalid BRIEF-A administrations")[1]
    assert "S95" in invalid_section
    assert "baseline" in invalid_section


def test_excluded_sessions_gated_on_the_trees_own_flag(two_trees):
    # main has exclude_low_accuracy=True (applied); supp has it False.
    report = build_report(two_trees)
    section = report.split("## Excluded sessions")[1].split("## Invalid BRIEF-A")[0]
    main_part, supp_part = section.split("**main**")[1].split("**supp_low_accuracy")

    assert "Rule not applied" not in main_part
    assert "excluded" in main_part and "not excluded" not in main_part

    assert "Rule not applied in this tree (exclude_low_accuracy=false)" in supp_part
    assert "not excluded" in supp_part
    assert "S97" in supp_part


def test_invalid_brief_gated_on_the_trees_own_flag(two_trees):
    main_root, supp_root = two_trees["main"], two_trees["supp"]
    supp_variant = json.loads((supp_root / "analysis_variant.json").read_text())
    supp_variant["exclude_invalid_brief"] = False
    (supp_root / "analysis_variant.json").write_text(json.dumps(supp_variant))

    report = build_report(two_trees)
    section = report.split("## Invalid BRIEF-A administrations")[1]
    main_part, supp_part = section.split("**main**")[1].split("**supp_low_accuracy")

    assert "Rule not applied" not in main_part
    assert "excluded" in main_part and "not excluded" not in main_part

    assert "Rule not applied in this tree (exclude_invalid_brief=false)" in supp_part
    assert "not excluded" in supp_part
    assert "S95" in supp_part


def test_excluded_sessions_empty_note_when_no_rows(two_trees, tmp_path):
    main_root, supp_root = two_trees["main"], two_trees["supp"]
    for root in (main_root, supp_root):
        path = root / "04b_session_info" / "low_accuracy_sessions.csv"
        pd.DataFrame(columns=["task", "subject", "session_id", "task_order",
                               "accuracy_metric", "accuracy"]).to_csv(path, index=False)

    report = build_report(two_trees)
    section = report.split("## Excluded sessions")[1].split("## Invalid BRIEF-A")[0]
    assert "no sessions at or below chance accuracy" in section


def test_cli_writes_report_file(two_trees, tmp_path):
    out = tmp_path / "report.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--trees", str(two_trees["main"]), str(two_trees["supp"]),
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.is_file()
    assert "Analysis variant comparison" in out.read_text()


def test_cli_rejects_duplicate_tree_paths(two_trees, tmp_path):
    out = tmp_path / "report.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--trees", str(two_trees["main"]), str(two_trees["main"]),
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "more than once" in result.stderr
    assert not out.exists()


def test_missing_main_tree_is_an_error(tmp_path):
    only_supp = tmp_path / "run_supp"
    _write_tree(only_supp, dict(MAIN_VARIANT, name="supp_low_accuracy", exclude_low_accuracy=False))

    with pytest.raises(SystemExit, match="main"):
        build_report({"supp": only_supp})
