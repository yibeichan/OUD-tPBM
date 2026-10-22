"""Cross-sectional hypothesis tests follow the run's analysis variant.

Covers three things:
- ``resolve_hypotheses`` swaps only H2's behavioral measure, per variant.
- ``global_performance_z`` survives one subject missing a task's metric
  (the bug: ``scipy.stats.zscore`` over a NaN-containing column returns an
  all-NaN array, so one missing metric used to blank the composite for
  everyone) and matches the old inline formula exactly when there is no NaN.
- ``load_behavioral_data`` skips sessions ``excluded_sessions`` names.

``unified_ef_model_common`` imports ``utils.statistical_methods``, which
exits the process if BASE_DIR/OUTPUT_DIR are unset at import time, and
``scripts/hypo_test`` carries no ``__init__.py`` (see
tests/test_longitudinal_followup_as_post.py for the same constraint on
scripts/longitudinal). tests/test_seeded_inference.py handles this by
importing once per module and popping shared modules (``seeding``,
``utils``) from ``sys.modules`` in teardown, but that is order-dependent:
a test file with no teardown discipline of its own could leave those
modules cached and bound to *this* file's tmp BASE_DIR/OUTPUT_DIR, which
would then leak into whichever test file imports them next. Running each
scenario in a subprocess sidesteps that entirely -- a fresh interpreter, so
nothing here is ever cached for another test file to inherit.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_HYPO_DIR = _REPO / "scripts" / "hypo_test"

_VARIANT_FIELDS = (
    "name", "channels", "exclude_low_accuracy", "h2_metric",
    "exclude_invalid_brief", "second_timepoint",
    "mask_wcst_without_regular_incorrect", "brain_direction",
)


def _run(base_dir, output_dir, body: str):
    """Run ``body`` in a subprocess with BASE_DIR/OUTPUT_DIR bound to the
    given directories and ``unified_ef_model_common`` importable as
    ``common``. Returns the completed process (stdout/stderr captured)."""
    preamble = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_REPO)!r})
        sys.path.insert(0, {str(_HYPO_DIR)!r})
        import unified_ef_model_common as common
        from scripts.analysis_variant import AnalysisVariant
    """)
    script = preamble + "\n" + textwrap.dedent(body)
    env = dict(os.environ)
    env["BASE_DIR"] = str(base_dir)
    env["OUTPUT_DIR"] = str(output_dir)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(_REPO),
    )


def test_resolve_swaps_only_h2(tmp_path):
    proc = _run(tmp_path, tmp_path, """
        base = {"H1": {"behavioral_measure": "interference"},
                "H2": {"behavioral_measure": "perseverative_errors"}}
        v = AnalysisVariant("x", "long", True, "perseverative_error_rate", True,
                            "most_complete", True, "efficiency")
        out = common.resolve_hypotheses(base, v)
        assert out["H2"]["behavioral_measure"] == "perseverative_error_rate"
        assert out["H1"] == base["H1"] and base["H2"]["behavioral_measure"] == "perseverative_errors"
    """)
    assert proc.returncode == 0, proc.stderr


def test_composite_survives_one_missing_task(tmp_path):
    proc = _run(tmp_path, tmp_path, """
        import numpy as np
        import pandas as pd
        # the bug this guards: zscore over a NaN-containing column returns all-NaN
        perf = pd.DataFrame({"subject_id": list("ABCD"),
                             "overall_accuracy": [0.8, np.nan, 0.9, 0.7],
                             "prop_rule_periods_mastered": [0.5, 0.6, 0.7, 0.4]})
        out = common.global_performance_z(perf)
        assert out.set_index("subject_id")["global_performance_z"].notna().tolist() == [True, True, True]
        assert "B" not in set(out.subject_id)
    """)
    assert proc.returncode == 0, proc.stderr


def test_global_performance_z_matches_old_formula_when_nan_free(tmp_path):
    proc = _run(tmp_path, tmp_path, """
        import numpy as np
        import pandas as pd
        from scipy.stats import zscore

        perf = pd.DataFrame({"subject_id": list("ABCD"),
                             "overall_accuracy": [0.8, 0.6, 0.9, 0.7],
                             "prop_rule_periods_mastered": [0.5, 0.6, 0.7, 0.4]})
        out = common.global_performance_z(perf)

        # The old inline formula, verbatim: same zscore call, same row order.
        old_stroop_z = zscore(perf["overall_accuracy"])
        old_wcst_z = zscore(perf["prop_rule_periods_mastered"])
        old = (old_stroop_z + old_wcst_z) / 2

        assert list(out["subject_id"]) == list(perf["subject_id"])
        assert np.allclose(out["global_performance_z"].values, old)
    """)
    assert proc.returncode == 0, proc.stderr


def _write_variant(output_dir: Path, **overrides):
    payload = {
        "name": "test",
        "channels": "all",
        "exclude_low_accuracy": True,
        "h2_metric": "perseverative_errors",
        "exclude_invalid_brief": False,
        "second_timepoint": "most_complete",
        "mask_wcst_without_regular_incorrect": False,
        "brain_direction": "efficiency",
    }
    payload.update(overrides)
    (output_dir / "analysis_variant.json").write_text(json.dumps(payload))


def _write_behavioral_tree(base_dir: Path, output_dir: Path, task, subj, task_id, accuracy_metric, accuracy):
    """One baseline session's task_info row + behavioral results.json."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    info_path = data_dir / f"{task}_task_info.csv"
    row = pd.DataFrame([{
        "filename": f"{subj}_{task_id}.snirf",
        "subj_id": subj,
        "task_id": task_id,
        "task_order": 1,
    }])
    if info_path.is_file():
        row = pd.concat([pd.read_csv(info_path), row], ignore_index=True)
    row.to_csv(info_path, index=False)

    behav_dir = output_dir / "02_behav" / task
    behav_dir.mkdir(parents=True, exist_ok=True)
    metrics = {accuracy_metric: accuracy}
    if task == "stroop":
        metrics.update({
            "overall_accuracy": accuracy, "stroop_effect_rt": 0.1, "mixing_cost_rt": 0.05,
            "overall_rt_cv": 0.2,
        })
    else:
        metrics.update({
            "overall_accuracy": accuracy, "categories_completed": 4,
            "prop_rule_periods_mastered": 0.5,
            "trials_to_first_category": 6,
            "perseverative_errors": 5, "perseverative_error_rate": 0.1,
            "wm_maintenance_decline": 0.3,
        })
    (behav_dir / f"{subj}_{task_id}_results.json").write_text(
        json.dumps({"metrics": metrics}))


def test_load_behavioral_data_skips_excluded_sessions(tmp_path):
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # S1 above chance (kept), S2 at chance (excluded when the variant asks).
    _write_behavioral_tree(base_dir, output_dir, "stroop", "S1", "s000001",
                            "overall_accuracy", 0.9)
    _write_behavioral_tree(base_dir, output_dir, "stroop", "S2", "s000002",
                            "overall_accuracy", 1 / 3)
    _write_behavioral_tree(base_dir, output_dir, "wcst", "S1", "w000001",
                            "exp_accuracy", 0.9)
    _write_variant(output_dir, exclude_low_accuracy=True)

    proc = _run(base_dir, output_dir, """
        stroop_df, wcst_df = common.load_behavioral_data()
        assert sorted(stroop_df["task_id"]) == ["s000001"]
        assert sorted(wcst_df["task_id"]) == ["w000001"]
    """)
    assert proc.returncode == 0, proc.stderr
    assert "Skipping Stroop session s000002" in proc.stdout


def test_load_behavioral_data_keeps_everything_when_variant_off(tmp_path):
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _write_behavioral_tree(base_dir, output_dir, "stroop", "S1", "s000001",
                            "overall_accuracy", 0.9)
    _write_behavioral_tree(base_dir, output_dir, "stroop", "S2", "s000002",
                            "overall_accuracy", 1 / 3)
    _write_behavioral_tree(base_dir, output_dir, "wcst", "S1", "w000001",
                            "exp_accuracy", 0.9)
    _write_variant(output_dir, exclude_low_accuracy=False)

    proc = _run(base_dir, output_dir, """
        stroop_df, wcst_df = common.load_behavioral_data()
        assert sorted(stroop_df["task_id"]) == ["s000001", "s000002"]
    """)
    assert proc.returncode == 0, proc.stderr
    assert "Skipping" not in proc.stdout
