"""Longitudinal loaders raise on missing inputs instead of losing levels.

There is no fallback to the baseline-only ROI file (the all-timepoints file
must cover every selected session), a missing behaviour JSON raises, a brain
delta needs the same ROI set at both timepoints, and subject IDs go through
standardize_subject_id. Both scripts are loaded by path; every fixture is
under tmp_path and session ids are synthetic.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "longitudinal"
_NAMES = ("unified_ef_longitudinal", "longitudinal_exploratory")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_guards", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=_NAMES)
def mod(request):
    return request.param, _load(request.param)


def _roi_loader(name, module):
    return module.load_roi_activations if name == "unified_ef_longitudinal" \
        else module.load_roi_brain_activation


def _behav_loader(name, module):
    return module.load_behavioral_metrics if name == "unified_ef_longitudinal" \
        else module.load_behavior_metrics


TASK_INFO = pd.DataFrame({
    "subject_id": ["S1", "S1"], "task_id": ["aaaaaaa", "bbbbbbb"], "task_order": [1, 2],
})


def _write_roi(out, sessions, subject="S1"):
    rows = [{"subject": subject, "session_id": s, "contrast": "stroop_effect",
             "network_roi": "Cont_PFCl", "activation": 0.1 * i}
            for i, s in enumerate(sessions)]
    for folder in ("10_longitudinal/roi_activations_alltimepoints/stroop", "06_roi_activation/stroop"):
        path = out / folder / "all_subjects_roi_activation_network_roi.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def test_all_timepoints_file_is_read(mod, tmp_path):
    name, module = mod
    _write_roi(tmp_path, ["aaaaaaa", "bbbbbbb"])
    df = _roi_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"], ["Cont_PFCl"], "stroop_effect")
    assert sorted(df["task_order"]) == [1, 2]


def test_no_fallback_to_the_baseline_only_file(mod, tmp_path):
    name, module = mod
    _write_roi(tmp_path, ["aaaaaaa", "bbbbbbb"])
    (tmp_path / "10_longitudinal" / "roi_activations_alltimepoints" / "stroop"
     / "all_subjects_roi_activation_network_roi.csv").unlink()
    with pytest.raises(FileNotFoundError, match="extract_roi_alltimepoints"):
        _roi_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"], ["Cont_PFCl"], "stroop_effect")


def test_incomplete_all_timepoints_file_raises(mod, tmp_path):
    name, module = mod
    _write_roi(tmp_path, ["aaaaaaa"])  # post session missing
    with pytest.raises(ValueError, match="bbbbbbb"):
        _roi_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"], ["Cont_PFCl"], "stroop_effect")


def test_non_canonical_subject_spelling_raises(mod, tmp_path):
    """A subject spelling that cannot be standardized raises, not an empty result."""
    name, module = mod
    _write_roi(tmp_path, ["aaaaaaa", "bbbbbbb"], subject="S1 extra")
    with pytest.raises(ValueError):
        _roi_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"], ["Cont_PFCl"], "stroop_effect")


def test_hyphenated_subject_is_standardized(mod, tmp_path):
    name, module = mod
    _write_roi(tmp_path, ["aaaaaaa", "bbbbbbb"], subject="S-1")
    df = _roi_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"], ["Cont_PFCl"], "stroop_effect")
    assert set(df["subject_id"]) == {"S1"}


def test_missing_behaviour_json_raises(mod, tmp_path):
    name, module = mod
    behav = tmp_path / "02_behav" / "stroop"
    behav.mkdir(parents=True)
    (behav / "S1_aaaaaaa_results.json").write_text(json.dumps({"metrics": {"overall_accuracy": 0.9}}))
    with pytest.raises(FileNotFoundError, match="S1_bbbbbbb_results.json"):
        _behav_loader(name, module)(tmp_path, TASK_INFO, "stroop", ["S1"])


def test_phq_subjects_are_standardized(tmp_path):
    """The exploratory PHQ-9/GAD-7 loader standardizes instead of rewriting."""
    module = _load("longitudinal_exploratory")
    scores = tmp_path / "01_assessments" / "scores"
    scores.mkdir(parents=True)
    pd.DataFrame({"subject": ["S-1", "S-1"], "timepoint": ["baseline", "post"],
                  "total_score": [7.0, 4.0]}).to_csv(scores / "phq9_scores.csv", index=False)
    result = module.load_assessment_scores(tmp_path, ["S1"], exclude_invalid_brief=False)
    assert sorted(result["value"]) == [4.0, 7.0]


_HYP = {
    "name": "Test", "self_report": "Test_t", "neural_contrast": "c",
    "neural_roi": ["ROI_A", "ROI_B"], "behavioral": "b", "task": "t",
    "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
    "description": "test",
}


def _run(module, roi_rows):
    assessment = pd.DataFrame({"subject_id": ["S1", "S1", "S2", "S2"], "metric": ["Test_t"] * 4,
                               "value": [60, 50, 40, 30], "task_order": [1, 2, 1, 2]})
    roi = pd.DataFrame(roi_rows, columns=["subject_id", "network_roi", "task_order", "activation"])
    roi["task"], roi["contrast"] = "t", "c"
    behavior = pd.DataFrame(columns=["subject_id", "task", "task_order", "timepoint", "metric", "value"])
    return module.run_hypothesis_analysis("HX", _HYP, assessment, behavior, roi,
                                          ["S1", "S2"], brain_direction=-1).set_index("subject_id")


def test_brain_delta_needs_the_same_roi_set():
    module = _load("unified_ef_longitudinal")
    rows = [("S1", "ROI_A", 1, 1.0), ("S1", "ROI_B", 1, 3.0), ("S1", "ROI_A", 2, 0.5),
            ("S2", "ROI_A", 1, 1.0), ("S2", "ROI_B", 1, 2.0),
            ("S2", "ROI_A", 2, 0.5), ("S2", "ROI_B", 2, 1.0)]
    results = _run(module, rows)
    assert np.isnan(results.loc["S1", "delta_brain"])
    assert results.loc["S1", "rci_brain_class"] == "insufficient_data"
    assert results.loc["S2", "delta_brain"] == pytest.approx(0.75 - 1.5)
