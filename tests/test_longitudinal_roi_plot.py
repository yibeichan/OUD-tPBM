"""The exploratory ROI trajectory figure is drawn per task.

The frame holds Stroop and WCST ROIs; one figure titled from its first row
showed the WCST ROIs under a Stroop title and merged the two Cont_PFCl
series. Synthetic frame under tmp_path.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def test_one_figure_per_task(module, tmp_path):
    rows = []
    for task, contrast, rois in [("stroop", "incongruent_vs_pure", ["Cont_PFCl", "SalVentAttn_Med"]),
                                 ("wcst", "learning", ["Cont_PFCl", "Cont_Par"])]:
        for roi in rois:
            for order, label in [(1, "baseline"), (2, "post")]:
                rows.append({"subject_id": "S1", "task": task, "contrast": contrast,
                             "network_roi": roi, "task_order": order, "timepoint": label,
                             "activation": 0.1 * order})
    module.generate_roi_trajectory_plot(pd.DataFrame(rows), "S1", tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "S1_stroop_brain_roi_trajectory.png", "S1_wcst_brain_roi_trajectory.png"]
