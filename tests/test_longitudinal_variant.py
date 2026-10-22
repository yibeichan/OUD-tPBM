"""Longitudinal classifier follows the run's analysis variant.

Covers three things:
- ``compute_global_performance_z`` drops a subject-timepoint unless both
  tasks contributed a z-score to it (the composite is a Stroop+WCST
  average; averaging over whichever task happens to be present would
  silently turn a one-task value into a two-task composite).
- ``resolve_longitudinal_hypotheses`` swaps only H2's behavioral metric,
  leaving ``HYPOTHESES`` itself untouched.
- ``run_hypothesis_analysis`` reads the neural domain's a-priori
  improvement direction from its ``brain_direction`` argument rather than
  a hard-coded ``-1``.

The module is loaded by path (``scripts/longitudinal/`` is not a package)
and calls ``load_dotenv()`` only inside ``main()``, so importing it touches
no configured data root; every fixture here is synthetic.
"""
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from analysis_variant import AnalysisVariant

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "longitudinal" / "unified_ef_longitudinal.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("unified_ef_longitudinal", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _variant(**overrides):
    payload = {
        "name": "test",
        "channels": "long",
        "exclude_low_accuracy": False,
        "h2_metric": "perseverative_errors",
        "exclude_invalid_brief": False,
        "second_timepoint": "most_complete",
        "mask_wcst_without_regular_incorrect": False,
        "brain_direction": "efficiency",
    }
    payload.update(overrides)
    return AnalysisVariant(**payload)


# =============================================================================
# compute_global_performance_z: both tasks required
# =============================================================================


def test_missing_stroop_at_one_timepoint_drops_only_that_row(module):
    # S2 has no Stroop row at all for task_order=2 (post); WCST has both.
    stroop_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2"],
        "task_order": [1, 2, 1],
        "timepoint": ["baseline", "post", "baseline"],
        "overall_accuracy": [0.9, 0.8, 0.7],
    })
    wcst_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"],
        "task_order": [1, 2, 1, 2],
        "timepoint": ["baseline", "post", "baseline", "post"],
        "prop_rule_periods_mastered": [0.5, 0.6, 0.4, 0.3],
    })

    result = module.compute_global_performance_z(stroop_df, wcst_df)

    present = set(zip(result["subject_id"], result["task_order"]))
    assert present == {("S1", 1), ("S1", 2), ("S2", 1)}
    assert ("S2", 2) not in present
    assert result["global_performance_z"].notna().all()


def test_both_tasks_present_matches_the_plain_mean(module):
    stroop_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"],
        "task_order": [1, 2, 1, 2],
        "timepoint": ["baseline", "post", "baseline", "post"],
        "overall_accuracy": [0.9, 0.8, 0.7, 0.6],
    })
    wcst_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"],
        "task_order": [1, 2, 1, 2],
        "timepoint": ["baseline", "post", "baseline", "post"],
        "prop_rule_periods_mastered": [0.5, 0.6, 0.4, 0.3],
    })

    result = module.compute_global_performance_z(stroop_df, wcst_df)

    assert len(result) == 4
    assert result["global_performance_z"].notna().all()


# =============================================================================
# resolve_longitudinal_hypotheses: H2 only, no mutation of HYPOTHESES
# =============================================================================


def test_resolve_swaps_only_h2(module):
    variant = _variant(h2_metric="perseverative_error_rate")

    resolved = module.resolve_longitudinal_hypotheses(module.HYPOTHESES, variant)

    assert resolved["H2"]["behavioral"] == "perseverative_error_rate"
    for hyp_id in ("H1", "H3", "H4", "H5"):
        assert resolved[hyp_id] == module.HYPOTHESES[hyp_id]
    # The module-level default is untouched by resolving against a variant.
    assert module.HYPOTHESES["H2"]["behavioral"] == "perseverative_errors"


def test_resolve_is_a_deep_copy(module):
    variant = _variant(h2_metric="perseverative_error_rate")
    resolved = module.resolve_longitudinal_hypotheses(module.HYPOTHESES, variant)
    resolved["H1"]["neural_roi"].append("bogus")
    assert "bogus" not in module.HYPOTHESES["H1"]["neural_roi"]


# =============================================================================
# run_hypothesis_analysis: brain_direction drives the convergence classification
# =============================================================================


_HYP_CONFIG = {
    "name": "Test",
    "self_report": "Test_t",
    "neural_contrast": "test_contrast",
    "neural_roi": ["ROI_A"],
    "behavioral": "test_behavior",
    "task": "test_task",
    "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
    "description": "test hypothesis",
}


def _synthetic_frames():
    # Self-report decreases (an a-priori improvement, direction fixed at -1
    # regardless of brain_direction) for both subjects.
    assessment_data = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"],
        "metric": ["Test_t"] * 4,
        "value": [60, 50, 40, 30],
        "task_order": [1, 2, 1, 2],
    })
    # Brain activation also decreases for both subjects.
    roi_data = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S2"],
        "task": ["test_task"] * 4,
        "contrast": ["test_contrast"] * 4,
        "network_roi": ["ROI_A"] * 4,
        "activation": [5, 3, 2, 1],
        "task_order": [1, 2, 1, 2],
    })
    behavior_data = pd.DataFrame(
        columns=["subject_id", "task", "task_order", "timepoint", "metric", "value"]
    )
    return assessment_data, behavior_data, roi_data


def test_brain_decrease_is_improvement_under_efficiency(module):
    assessment_data, behavior_data, roi_data = _synthetic_frames()

    results = module.run_hypothesis_analysis(
        "HX", _HYP_CONFIG, assessment_data, behavior_data, roi_data,
        ["S1", "S2"], brain_direction=-1,
    )

    # Self-report and brain both move toward improvement -> fully convergent.
    assert set(results["convergence_class"]) == {"convergent_improvement"}


def test_brain_decrease_is_divergence_under_activation(module):
    assessment_data, behavior_data, roi_data = _synthetic_frames()

    results = module.run_hypothesis_analysis(
        "HX", _HYP_CONFIG, assessment_data, behavior_data, roi_data,
        ["S1", "S2"], brain_direction=1,
    )

    # Self-report improves but brain now counts as moving away from
    # improvement -> the two domains cancel out.
    assert set(results["convergence_class"]) == {"mixed"}
