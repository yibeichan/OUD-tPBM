"""compare_run_trees.py diffs two run trees stage by stage.

All fixtures are tmp_path trees; no real run tree is touched.
"""
import json
import subprocess
import sys
from pathlib import Path

from scripts.verification.compare_run_trees import compare_run_trees

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verification" / "compare_run_trees.py"


def _tree(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    return root


def test_identical_trees_report_no_difference(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "05_channel_group_analysis").mkdir()
        (root / "05_channel_group_analysis" / "results.csv").write_text("subject,value\nS1,1.0\nS2,2.0\n")

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert has_diff is False
    assert any("byte-equal" in line and "results.csv" in line for line in lines)


def test_missing_file_is_named_and_flags_difference(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (a / "05_channel_group_analysis" / "results.csv").write_text("subject,value\nS1,1.0\n")
    (b / "05_channel_group_analysis").mkdir()

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert has_diff is True
    assert any(line == "A-only: 05_channel_group_analysis/results.csv" for line in lines)


def test_changed_cell_is_named_with_max_diff_and_count(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis").mkdir()
    (a / "05_channel_group_analysis" / "results.csv").write_text(
        "subject,value\nS1,1.0\nS2,2.0\nS3,3.0\n"
    )
    (b / "05_channel_group_analysis" / "results.csv").write_text(
        "subject,value\nS1,1.0\nS2,2.5\nS3,3.0\n"
    )

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert has_diff is True
    diff_line = next(line for line in lines if "results.csv" in line)
    assert diff_line.startswith("different")
    assert "value: max|diff|=0.5, 1 cell(s) differ" in diff_line


def test_ignored_csv_column_makes_files_equal(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "10_longitudinal").mkdir()
    (b / "10_longitudinal").mkdir()
    (a / "10_longitudinal" / "hypothesis_convergence.csv").write_text(
        "subject,convergence_index,variant\nS1,0.5,main\n"
    )
    (b / "10_longitudinal" / "hypothesis_convergence.csv").write_text(
        "subject,convergence_index,variant\nS1,0.5,supp_count\n"
    )

    has_diff, lines = compare_run_trees(a, b, stages=["10"])

    assert has_diff is False
    assert any(line.startswith("equal-after-ignores") for line in lines)


def _h2_trees(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for tree, rate in ((a, "0.2"), (b, "0.9")):
        (tree / "10_longitudinal").mkdir()
        (tree / "10_longitudinal" / "merged_data.csv").write_text(
            f"subject,perseverative_error_rate\nS1,{rate}\n"
        )
        (tree / "10_longitudinal" / "summary.json").write_text(
            json.dumps({"h2_measure": rate, "variant": tree.name})
        )
    return a, b


def test_default_ignores_compare_result_columns_and_keys(tmp_path):
    a, b = _h2_trees(tmp_path)

    has_diff, lines = compare_run_trees(a, b, stages=["10"])

    assert has_diff is True
    assert any(l.startswith("different") and "merged_data.csv" in l for l in lines)
    assert any(l.startswith("different") and "summary.json" in l for l in lines)


def test_result_columns_and_keys_are_ignored_only_when_named(tmp_path):
    a, b = _h2_trees(tmp_path)

    has_diff, lines = compare_run_trees(
        a, b, stages=["10"],
        ignore_csv_columns=["variant", "perseverative_error_rate"],
        ignore_json_keys=["variant", "h2_measure"],
    )

    assert has_diff is False


def test_cli_ignore_flag_with_no_values_ignores_nothing(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for tree in (a, b):
        (tree / "10_longitudinal").mkdir()
        (tree / "10_longitudinal" / "t.csv").write_text(
            f"subject,variant\nS1,{tree.name}\n"
        )

    default = subprocess.run([sys.executable, str(SCRIPT), str(a), str(b), "--stages", "10"],
                             capture_output=True, text=True)
    none = subprocess.run([sys.executable, str(SCRIPT), str(a), str(b), "--stages", "10",
                           "--ignore-csv-columns"], capture_output=True, text=True)

    assert default.returncode == 0, default.stdout + default.stderr
    assert none.returncode == 1, none.stdout + none.stderr


def test_ignored_json_key_makes_files_equal(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "09_hypothesis_testing").mkdir()
    (b / "09_hypothesis_testing").mkdir()
    (a / "09_hypothesis_testing" / "results.json").write_text(
        json.dumps({"generated_at": "2026-09-22T00:00:00", "value": 1.0})
    )
    (b / "09_hypothesis_testing" / "results.json").write_text(
        json.dumps({"generated_at": "2026-09-26T00:00:00", "value": 1.0})
    )

    has_diff, lines = compare_run_trees(a, b, stages=["09"])

    assert has_diff is False
    assert any(line.startswith("equal-after-ignores") for line in lines)


def test_json_difference_beyond_ignored_keys_is_flagged(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "09_hypothesis_testing").mkdir()
    (b / "09_hypothesis_testing").mkdir()
    (a / "09_hypothesis_testing" / "results.json").write_text(
        json.dumps({"generated_at": "t1", "value": 1.0})
    )
    (b / "09_hypothesis_testing" / "results.json").write_text(
        json.dumps({"generated_at": "t2", "value": 2.0})
    )

    has_diff, lines = compare_run_trees(a, b, stages=["09"])

    assert has_diff is True
    diff_line = next(line for line in lines if "results.json" in line)
    assert diff_line.startswith("different")
    assert "value" in diff_line


def test_png_is_listed_but_not_compared(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "08_glm_topoplots").mkdir()
    (b / "08_glm_topoplots").mkdir()
    (a / "08_glm_topoplots" / "fig.png").write_bytes(b"not-a-real-png-a")
    (b / "08_glm_topoplots" / "fig.png").write_bytes(b"not-a-real-png-b")

    has_diff, lines = compare_run_trees(a, b, stages=["08"])

    assert has_diff is False
    assert any(line == "skipped (binary): 08_glm_topoplots/fig.png" for line in lines)


def test_stages_matches_two_digit_prefix(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "05_channel_group_analysis").mkdir()
        (root / "05_channel_group_analysis" / "x.csv").write_text("a\n1\n")
        (root / "10_longitudinal").mkdir()
        (root / "10_longitudinal" / "y.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert not any("10_longitudinal" in line for line in lines)
    assert any("05_channel_group_analysis" in line for line in lines)


def test_only_overrides_stages_and_takes_exact_relpath(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "10_longitudinal" / "unified_ef").mkdir(parents=True)
        (root / "10_longitudinal" / "unified_ef" / "hypothesis_convergence.csv").write_text("a\n1\n")
        (root / "10_longitudinal" / "other.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, only=["10_longitudinal/unified_ef"])

    assert has_diff is False
    assert any("hypothesis_convergence.csv" in line for line in lines)
    assert not any("other.csv" in line for line in lines)


def test_exclude_drops_a_subpath_from_the_stage_walk(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "10_longitudinal" / "unified_ef").mkdir(parents=True)
    (b / "10_longitudinal" / "unified_ef").mkdir(parents=True)
    (a / "10_longitudinal" / "unified_ef" / "hypothesis_convergence.csv").write_text("a\n1\n")
    (b / "10_longitudinal" / "unified_ef" / "hypothesis_convergence.csv").write_text("a\n2\n")
    (a / "10_longitudinal" / "shared.csv").write_text("a\n1\n")
    (b / "10_longitudinal" / "shared.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, stages=["10"], exclude=["10_longitudinal/unified_ef"])

    assert has_diff is False
    assert not any("hypothesis_convergence.csv" in line for line in lines)
    assert any("shared.csv" in line for line in lines)


def test_follows_symlinks(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    real_dir = tmp_path / "elsewhere" / "10_longitudinal"
    real_dir.mkdir(parents=True)
    (real_dir / "data.csv").write_text("a\n1\n")
    (a / "10_longitudinal").symlink_to(real_dir, target_is_directory=True)
    (b / "10_longitudinal").mkdir()
    (b / "10_longitudinal" / "data.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, stages=["10"])

    assert has_diff is False
    assert any("data.csv" in line and "byte-equal" in line for line in lines)


def test_cli_exits_zero_on_no_difference(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "05_channel_group_analysis").mkdir()
        (root / "05_channel_group_analysis" / "x.csv").write_text("a\n1\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), "--stages", "05"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_typo_stage_prefix_matching_neither_tree_errors(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis").mkdir()

    has_diff, lines = compare_run_trees(a, b, stages=["5"])

    assert has_diff is True
    assert any("'5'" in line and "ERROR" in line for line in lines)


def test_valid_and_typo_stage_prefixes_together_still_error(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "05_channel_group_analysis").mkdir()
        (root / "05_channel_group_analysis" / "x.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, stages=["05", "99"])

    assert has_diff is True
    assert any("'99'" in line and "ERROR" in line for line in lines)
    # the valid prefix's files are still compared
    assert any(line.startswith("byte-equal") and "x.csv" in line for line in lines)


def test_only_path_matching_neither_tree_errors(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")

    has_diff, lines = compare_run_trees(a, b, only=["10_longitudinal/does_not_exist"])

    assert has_diff is True
    assert any("does_not_exist" in line and "ERROR" in line for line in lines)


def test_only_path_matching_one_side_is_not_flagged_as_unmatched(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "10_longitudinal" / "unified_ef").mkdir(parents=True)
    (a / "10_longitudinal" / "unified_ef" / "f.csv").write_text("a\n1\n")

    has_diff, lines = compare_run_trees(a, b, only=["10_longitudinal/unified_ef"])

    assert has_diff is True
    assert not any("ERROR" in line for line in lines)
    assert any(line == "A-only: 10_longitudinal/unified_ef/f.csv" for line in lines)


def test_summary_line_reports_file_counts(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    for root in (a, b):
        (root / "05_channel_group_analysis").mkdir()
        (root / "05_channel_group_analysis" / "x.csv").write_text("a\n1\n")

    _, lines = compare_run_trees(a, b, stages=["05"])

    assert lines[-1] == (
        "compared 1 file(s): 1 byte-equal, 0 equal-after-ignores, "
        "0 different, 0 skipped (binary), 0 one-sided"
    )


def test_zero_file_run_is_visible_in_summary_and_not_an_error(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis").mkdir()

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert has_diff is False
    assert not any("ERROR" in line for line in lines)
    assert lines[-1] == (
        "compared 0 file(s): 0 byte-equal, 0 equal-after-ignores, "
        "0 different, 0 skipped (binary), 0 one-sided"
    )


def test_column_order_alone_is_equal_after_ignores(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis").mkdir()
    (a / "05_channel_group_analysis" / "x.csv").write_text("subject,value\nS1,1.0\n")
    (b / "05_channel_group_analysis" / "x.csv").write_text("value,subject\n1.0,S1\n")

    has_diff, lines = compare_run_trees(a, b, stages=["05"])

    assert has_diff is False
    assert any(line.startswith("equal-after-ignores") and "x.csv" in line for line in lines)


def test_cli_exits_one_on_typo_stage_prefix(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis").mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), "--stages", "5"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ERROR" in result.stdout


def test_cli_exits_one_on_unmatched_only_path(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), "--only", "no/such/path"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ERROR" in result.stdout


def test_cli_exits_one_on_difference(tmp_path):
    a, b = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    (a / "05_channel_group_analysis").mkdir()
    (a / "05_channel_group_analysis" / "x.csv").write_text("a\n1\n")
    (b / "05_channel_group_analysis").mkdir()
    (b / "05_channel_group_analysis" / "x.csv").write_text("a\n2\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(a), str(b), "--stages", "05"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
