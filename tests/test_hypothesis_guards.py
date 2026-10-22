"""Hypothesis-testing inputs that are missing or degenerate raise, not shrink.

The stage modules bind OUTPUT_DIR/BASE_DIR at import, so they are imported
once against a throwaway directory and module globals are pointed at each
test's own tree. Session ids here are synthetic.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning

_REPO = Path(__file__).resolve().parents[1]
_MODULES = ("roi_utils", "unified_ef_model_common", "unified_ef_model_roi",
            "unified_ef_model_channel", "utils.statistical_methods", "utils", "seeding")


@pytest.fixture(scope="module")
def mods(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_REPO / "scripts" / "brain"))
        mp.syspath_prepend(str(_REPO / "scripts" / "hypo_test"))
        yield {
            "common": importlib.import_module("unified_ef_model_common"),
            "roi": importlib.import_module("unified_ef_model_roi"),
            "channel": importlib.import_module("unified_ef_model_channel"),
            "stats": importlib.import_module("utils.statistical_methods"),
        }
    for name in _MODULES:
        sys.modules.pop(name, None)


# --------------------------------------------------------------------------
# A constant variable is untestable, not maximally significant
# --------------------------------------------------------------------------

def test_constant_input_is_insufficient_data_roi(mods):
    x = np.full(6, 60.0)
    y = np.arange(6, dtype=float)
    # The warning is the only other signal, so it must stay visible.
    with pytest.warns(ConstantInputWarning):
        result = mods["common"].test_correlation(x, y, "t", n_perm=99, n_boot=20)
    assert result["status"] == "insufficient_data"
    assert np.isnan(result["p"])


def test_constant_input_is_insufficient_data_channel(mods):
    x = np.arange(6, dtype=float)
    y = np.full(6, 0.25)
    with pytest.warns(ConstantInputWarning):
        result = mods["channel"].test_correlation_fast(x, y, n_perm=99, n_boot=20)
    assert result["status"] == "insufficient_data"
    assert np.isnan(result["p"])


def test_varying_input_still_completes(mods):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=8), rng.normal(size=8)
    assert mods["common"].test_correlation(x, y, "t", n_perm=99, n_boot=20)["status"] == "completed"
    assert mods["channel"].test_correlation_fast(x, y, n_perm=99, n_boot=20)["status"] == "completed"


# --------------------------------------------------------------------------
# The confirmatory family is always 8
# --------------------------------------------------------------------------

def _family(n_primary):
    tests = []
    names = [f"H{h}_{kind}" for h in (1, 2, 3, 4) for kind in ("self_brain", "brain_behav")]
    for name in names[:n_primary]:
        tests.append({"test_name": name, "status": "completed", "p": 0.2})
    return {"H1": {"tests": tests}, "H5": {"tests": [
        {"test_name": "H5_global_performance_z", "status": "completed", "p": 0.3}]}}


def test_primary_family_of_eight_passes(mods):
    roi = mods["roi"]
    results = roi.apply_fdr_correction(_family(8))
    roi.check_primary_family(results)


def test_shrunken_primary_family_raises(mods):
    roi = mods["roi"]
    results = roi.apply_fdr_correction(_family(4))
    with pytest.raises(ValueError, match="has 4 tests, expected 8"):
        roi.check_primary_family(results)


# --------------------------------------------------------------------------
# A subject missing one ROI of a two-ROI pathway
# --------------------------------------------------------------------------

def _roi_rows(rows):
    return pd.DataFrame(rows, columns=["subject", "contrast", "network_roi",
                                       "activation", "t_statistic", "n_channels"])


def test_two_roi_pathway_requires_both_rois(mods):
    roi_df = _roi_rows([
        ("S1", "stroop_effect", "Cont_PFCl", 0.1, 1.0, 5),
        ("S1", "stroop_effect", "SalVentAttn_PFCl", 0.3, 1.0, 5),
        ("S2", "stroop_effect", "Cont_PFCl", 0.2, 1.0, 5),
    ])
    with pytest.raises(ValueError, match=r"\['S2'\] lack one of the ROIs"):
        mods["roi"].aggregate_roi_activation(roi_df, ["SalVentAttn_PFCl", "Cont_PFCl"], "stroop_effect")


def test_two_roi_pathway_with_both_rois_averages(mods):
    roi_df = _roi_rows([
        ("S1", "stroop_effect", "Cont_PFCl", 0.1, 1.0, 5),
        ("S1", "stroop_effect", "SalVentAttn_PFCl", 0.3, 1.0, 5),
    ])
    agg = mods["roi"].aggregate_roi_activation(roi_df, ["SalVentAttn_PFCl", "Cont_PFCl"], "stroop_effect")
    assert agg["activation"].tolist() == pytest.approx([0.2])


# --------------------------------------------------------------------------
# A missing measure column is a KeyError, not a fuzzy substitute
# --------------------------------------------------------------------------

def test_missing_behaviour_column_raises(mods):
    roi = mods["roi"]
    hyp = dict(roi.PRIMARY_HYPOTHESES["H1"])
    merged = pd.DataFrame({"subject": ["S1", "S2"], "Inhibit_t": [60, 70],
                           "interference_ms": [0.1, 0.2]})
    roi_df = _roi_rows([
        (s, "stroop_effect", r, 0.1, 1.0, 5)
        for s in ("S1", "S2") for r in ("Cont_PFCl", "SalVentAttn_PFCl")
    ])
    data = {"merged": merged, "roi_stroop": roi_df}
    with pytest.raises(KeyError, match="interference"):
        roi.test_hypothesis_roi_level("H1", hyp, data)


def test_missing_h5_column_raises(mods):
    roi = mods["roi"]
    merged = pd.DataFrame({"subject": ["S1"], "GEC_total_t": [60],
                           "global_performance_z": [0.1]})
    with pytest.raises(KeyError, match="GEC_t"):
        roi.test_hypothesis_roi_level("H5", roi.PRIMARY_HYPOTHESES["H5"], {"merged": merged})


# --------------------------------------------------------------------------
# The behaviour loader raises on a missing file, a missing metrics
# block or a missing metric instead of skipping or substituting
# --------------------------------------------------------------------------

def _behav_tree(tmp_path, stats, monkeypatch, wcst_metrics=None, drop_stroop=False, stroop_payload=None):
    base, out = tmp_path / "base", tmp_path / "out"
    (base / "data").mkdir(parents=True)
    cols = ["filename", "subj_id", "task_id", "task_order"]
    for task in ("stroop", "wcst"):
        pd.DataFrame([(f"{task}.snirf", "S1", f"{task[0]}aaaaaa", 1)], columns=cols).to_csv(
            base / "data" / f"{task}_task_info.csv", index=False)
        (out / "02_behav" / task).mkdir(parents=True)
    if not drop_stroop:
        payload = stroop_payload if stroop_payload is not None else {"metrics": {"overall_accuracy": 0.9}}
        (out / "02_behav" / "stroop" / "S1_saaaaaa_results.json").write_text(json.dumps(payload))
    metrics = wcst_metrics if wcst_metrics is not None else {
        "categories_completed": 5, "prop_rule_periods_mastered": 0.5,
        "trials_to_first_category": 7}
    (out / "02_behav" / "wcst" / "S1_waaaaaa_results.json").write_text(json.dumps({"metrics": metrics}))
    monkeypatch.setattr(stats, "BASE_DIR", str(base))
    monkeypatch.setattr(stats, "BEHAV_DIR", out / "02_behav")
    monkeypatch.setattr(stats, "excluded_sessions", lambda task: set())


def test_behaviour_loader_reads_a_complete_tree(mods, tmp_path, monkeypatch):
    stats = mods["stats"]
    _behav_tree(tmp_path, stats, monkeypatch)
    stroop_df, wcst_df = stats.load_behavioral_data()
    assert stroop_df["subject_id"].tolist() == ["S1"]
    assert wcst_df["categories_completed"].tolist() == [5]
    assert wcst_df["prop_rule_periods_mastered"].tolist() == [0.5]


def test_missing_behaviour_file_raises(mods, tmp_path, monkeypatch):
    stats = mods["stats"]
    _behav_tree(tmp_path, stats, monkeypatch, drop_stroop=True)
    with pytest.raises(FileNotFoundError, match="S1_saaaaaa_results.json"):
        stats.load_behavioral_data()


def test_behaviour_file_without_metrics_raises(mods, tmp_path, monkeypatch):
    stats = mods["stats"]
    _behav_tree(tmp_path, stats, monkeypatch, stroop_payload={"trials": []})
    with pytest.raises(KeyError, match="metrics"):
        stats.load_behavioral_data()


def test_missing_categories_completed_is_not_substituted(mods, tmp_path, monkeypatch):
    stats = mods["stats"]
    _behav_tree(tmp_path, stats, monkeypatch,
                wcst_metrics={"total_rule_changes": 9, "trials_to_first_category": 7,
                              "prop_rule_periods_mastered": 0.5})
    with pytest.raises(KeyError, match="categories_completed"):
        stats.load_behavioral_data()


def test_missing_h5_wcst_metric_is_not_substituted(mods, tmp_path, monkeypatch):
    # H5 reads the proportion of rule periods mastered; a results
    # file without it must not fall back to the count.
    stats = mods["stats"]
    _behav_tree(tmp_path, stats, monkeypatch,
                wcst_metrics={"categories_completed": 5, "trials_to_first_category": 7})
    with pytest.raises(KeyError, match="prop_rule_periods_mastered"):
        stats.load_behavioral_data()
