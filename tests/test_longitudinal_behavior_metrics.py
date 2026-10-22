"""Tests that every behavioural metric the longitudinal classifier asks for exists.

``HYPOTHESES`` names H1 and H4's behavioural arms by construct
(``interference``, ``rt_variability``), but the per-session results JSON
stores them under their computed names (``stroop_effect_rt``,
``overall_rt_cv``). The cross-sectional loader maps one to the other; the
longitudinal loader did not, so the classifier's metric filter matched no
row and H1 and H4 were classified from two domains instead of three for
every participant, without an error or a warning.

The module is loaded by path (``scripts/longitudinal/`` is not a package) and
calls ``load_dotenv()`` only inside ``main()``, so importing it touches no
configured data root; every fixture here is under ``tmp_path``.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "longitudinal" / "unified_ef_longitudinal.py")


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("unified_ef_longitudinal", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Keys as 01_stroop_behav.py and 01_wcst_behav.py write them.
_COMPUTED_METRICS = {
    "stroop": {"stroop_effect_rt": 0.1377, "overall_rt_cv": 0.185,
               "overall_accuracy": 0.95, "mixing_cost_rt": 0.040},
    "wcst": {"perseverative_errors": 20, "wm_maintenance_decline": 0.1,
             "categories_completed": 5},
}


def _write_session(output_dir, task, subject, task_id, metrics):
    task_dir = output_dir / "02_behav" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{subject}_{task_id}_results.json").write_text(
        json.dumps({"metrics": metrics}))


@pytest.mark.parametrize("hypothesis", ["H1", "H2", "H3", "H4"])
def test_behavioral_metric_is_loaded_for_each_hypothesis(module, tmp_path, hypothesis):
    config = module.HYPOTHESES[hypothesis]
    task = config["task"]
    _write_session(tmp_path, task, "S1", "abc1234", _COMPUTED_METRICS[task])
    info = pd.DataFrame({"subject_id": ["S1"], "task_id": ["abc1234"], "task_order": [1]})

    loaded = module.load_behavioral_metrics(tmp_path, info, task, ["S1"])

    assert config["behavioral"] in loaded.columns, (
        f"{hypothesis} asks for {config['behavioral']!r}, which the loader "
        f"does not produce; loaded columns: {sorted(loaded.columns)}")


@pytest.mark.parametrize("construct, computed", [
    ("interference", "stroop_effect_rt"),
    ("rt_variability", "overall_rt_cv"),
])
def test_construct_name_carries_the_computed_value(module, tmp_path, construct, computed):
    _write_session(tmp_path, "stroop", "S1", "abc1234", _COMPUTED_METRICS["stroop"])
    info = pd.DataFrame({"subject_id": ["S1"], "task_id": ["abc1234"], "task_order": [1]})

    loaded = module.load_behavioral_metrics(tmp_path, info, "stroop", ["S1"])

    assert loaded[construct].iloc[0] == _COMPUTED_METRICS["stroop"][computed]
    # The computed name stays too: subject_changes.csv and the §5.5.1 tables use it.
    assert loaded[computed].iloc[0] == _COMPUTED_METRICS["stroop"][computed]


def test_missing_computed_metric_stays_missing(module, tmp_path):
    """An alias must not invent a value when its source metric is absent."""
    metrics = {k: v for k, v in _COMPUTED_METRICS["stroop"].items() if k != "overall_rt_cv"}
    _write_session(tmp_path, "stroop", "S1", "abc1234", metrics)
    info = pd.DataFrame({"subject_id": ["S1"], "task_id": ["abc1234"], "task_order": [1]})

    loaded = module.load_behavioral_metrics(tmp_path, info, "stroop", ["S1"])

    assert "rt_variability" not in loaded.columns or loaded["rt_variability"].isna().all()
