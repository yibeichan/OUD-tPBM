"""step1's per-contrast bar helper must draw, not fall into its error handler."""
import importlib.util
import logging
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"


def _load(name, tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(BRAIN))
    spec = importlib.util.spec_from_file_location(name, BRAIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maps():
    rows = {
        i: {"ch_name": f"S{i}_D{i} {hb}", "effect": 0.1 * i, "se": 0.05, "p_value": 0.01}
        for i, hb in ((1, "hbo"), (2, "hbo"), (3, "hbr"))
    }
    return {"c": rows}


@pytest.mark.parametrize("module_name,cls_name", [
    ("step1_glm_stroop", "StroopfNIRSAnalysis"),
    ("step1_glm_wcst", "WCSTfNIRSAnalysis"),
])
def test_plot_contrast_subset_draws_without_error(module_name, cls_name, tmp_path, monkeypatch, caplog):
    module = _load(module_name, tmp_path, monkeypatch)
    analysis = getattr(module, cls_name).__new__(getattr(module, cls_name))
    analysis.contrast_maps = _maps()
    df = pd.DataFrame.from_dict(analysis.contrast_maps["c"], orient="index")
    hbo = df[df["ch_name"].str.contains("hbo")]

    fig, ax = plt.subplots()
    params = module_name == "step1_glm_stroop"
    with caplog.at_level(logging.ERROR):
        if params:
            analysis._plot_contrast_subset(ax, "c", hbo, "hbo", "r")
        else:
            analysis._plot_contrast_subset(ax, "c", "hbo", "r")
    plt.close(fig)

    assert "Error" not in ax.get_title()
    assert len(ax.patches) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    sys.modules.pop(module_name, None)
