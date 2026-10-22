"""H5's WCST component is the proportion of rule periods mastered.

``categories_completed`` counts rule periods in which three consecutive
correct responses were reached. How many periods a participant reaches
depends on how many trials they got through (86-144 at baseline), so the
count tracks trials completed (Spearman .66). H5 now z-scores categories
completed / rule periods; the count stays a reported metric.
"""
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_BEHAV = _REPO / "scripts" / "behav" / "01_wcst_behav.py"
_LONG = _REPO / "scripts" / "longitudinal" / "unified_ef_longitudinal.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(n, rule, change, outcome):
    return {
        "timestamp": 10.0 + 3 * n, "event": "start_trial", "block_type": "experimental",
        "trial": n, "trial_type": rule, "rule_change": "yes" if change else "no",
        "response_outcome": outcome, "response_pressed": 9.0, "response_RT": 1.0,
        "response_correct_streak": 0, "stimulus_ambiguity_level": 1,
        "stimulus_card_color": "blue", "stimulus_card_shape": "star",
        "stimulus_card_number": 2.0,
    }


def test_behaviour_reports_the_proportion_and_keeps_the_count():
    behav = _load(_BEHAV, "wcst_behav")
    # Four rule periods of five trials; the third never reaches 3 in a row.
    outcomes = {
        "color": [1, 1, 1, 1, 1],
        "shape": [0, 1, 1, 1, 0],
        "number": [1, 0, 1, 0, 1],
        "color2": [0, 0, 1, 1, 1],
    }
    rows, n = [], 1
    for rule, outs in outcomes.items():
        for i, outcome in enumerate(outs):
            rows.append(_trial(n, rule.rstrip("2"), i == 0, outcome))
            n += 1
    metrics = behav.WCSTEventAnalyzer(pd.DataFrame(rows)).get_performance_metrics()

    assert metrics["categories_completed"] == 3
    assert metrics["rule_periods"] == 4
    assert metrics["prop_rule_periods_mastered"] == pytest.approx(0.75)


def test_cross_sectional_composite_uses_the_proportion(tmp_path):
    # Counts and proportions rank the four subjects in opposite orders, so
    # the composite shows which one it z-scored.
    script = textwrap.dedent(f"""
        import sys
        import numpy as np
        import pandas as pd
        from scipy.stats import zscore
        sys.path.insert(0, {str(_REPO)!r})
        sys.path.insert(0, {str(_REPO / "scripts" / "hypo_test")!r})
        import unified_ef_model_common as common

        assert common.H5_WCST_METRIC == "prop_rule_periods_mastered"
        perf = pd.DataFrame({{"subject_id": list("ABCD"),
                              "overall_accuracy": [0.8, 0.6, 0.9, 0.7],
                              "categories_completed": [14, 12, 10, 8],
                              "prop_rule_periods_mastered": [0.4, 0.5, 0.6, 0.7]}})
        out = common.global_performance_z(perf)
        expected = (zscore(perf["overall_accuracy"]) + zscore(perf["prop_rule_periods_mastered"])) / 2
        assert np.allclose(out["global_performance_z"].values, expected)
    """)
    env = dict(os.environ, BASE_DIR=str(tmp_path), OUTPUT_DIR=str(tmp_path))
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, cwd=str(_REPO))
    assert proc.returncode == 0, proc.stderr


def test_longitudinal_composite_uses_the_proportion():
    module = _load(_LONG, "unified_ef_longitudinal")
    stroop_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"], "task_order": [1, 2, 1, 2],
        "timepoint": ["baseline", "post", "baseline", "post"],
        "overall_accuracy": [0.9, 0.9, 0.7, 0.7],
    })
    # The count rises for S1 while its proportion falls.
    wcst_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"], "task_order": [1, 2, 1, 2],
        "timepoint": ["baseline", "post", "baseline", "post"],
        "categories_completed": [5, 9, 4, 4],
        "prop_rule_periods_mastered": [0.8, 0.6, 0.5, 0.5],
    })
    result = module.compute_global_performance_z(stroop_df, wcst_df).set_index(
        ["subject_id", "task_order"])["global_performance_z"]
    assert result[("S1", 2)] < result[("S1", 1)]


def test_longitudinal_composite_raises_without_the_proportion():
    """Review m2: stale results without the proportion must not drop H5's WCST arm."""
    module = _load(_LONG, "unified_ef_longitudinal")
    stroop_df = pd.DataFrame({"subject_id": ["S1", "S2"], "task_order": [1, 1],
                              "timepoint": ["baseline"] * 2, "overall_accuracy": [0.9, 0.7]})
    wcst_df = pd.DataFrame({"subject_id": ["S1", "S2"], "task_order": [1, 1],
                            "timepoint": ["baseline"] * 2, "categories_completed": [5, 4]})
    with pytest.raises(KeyError, match="prop_rule_periods_mastered"):
        module.compute_global_performance_z(stroop_df, wcst_df)
