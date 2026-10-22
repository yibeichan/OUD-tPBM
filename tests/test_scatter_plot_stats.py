"""Scatter-plot annotations print the reported test, not a recomputation.

create_scatter_plot used to recompute r/p/CI with 5,000 permutations and
1,000 bootstraps, so the plotted p and CI differed from results_summary.csv
(10,000 / 5,000) in the third decimal. Synthetic data.
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODULES = ("roi_utils", "unified_ef_model_common", "unified_ef_model_roi",
            "unified_ef_model_channel", "utils.statistical_methods", "utils", "seeding")


@pytest.fixture(scope="module")
def common(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_REPO / "scripts" / "brain"))
        mp.syspath_prepend(str(_REPO / "scripts" / "hypo_test"))
        yield importlib.import_module("unified_ef_model_common")
    for name in _MODULES:
        sys.modules.pop(name, None)


def _test_df():
    rng = np.random.default_rng(3)
    return pd.DataFrame({"subject": [f"S{i}" for i in range(1, 9)],
                         "Inhibit_t": rng.normal(60, 8, 8),
                         "activation": rng.normal(0, 1, 8),
                         "interference": rng.normal(0.1, 0.03, 8)})


def _capture(monkeypatch):
    import utils.statistical_methods as sm
    captured = {}
    real = sm.create_scatter_plot

    def spy(x, y, xl, yl, subjects, title="", save_path=None, *, rng=None, stats=None):
        captured[Path(save_path).stem] = stats
        return real(x, y, xl, yl, subjects, title, None, rng=rng, stats=stats)

    monkeypatch.setattr(sm, "create_scatter_plot", spy)
    return captured


HYPO = {"name": "Inhibition"}


def test_roi_plots_print_the_reported_tests(common, monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    tests = [{"test_name": f"H1_{k}", "r": 0.111, "p": 0.222, "ci_lower": -0.3,
              "ci_upper": 0.4, "n": 8} for k in ("self_brain", "brain_behav", "self_behav")]
    common.create_hypothesis_scatter_plots(
        "H1", HYPO, _test_df(), "Inhibit_t", "interference", "stroop_effect",
        tmp_path, level="roi", tests=tests, seed=20260922)
    assert set(captured) == {"H1_self_brain", "H1_brain_behav", "H1_self_behav"}
    assert all(s["p"] == 0.222 for s in captured.values())


def test_without_reported_tests_the_roi_test_is_reproduced(common, monkeypatch, tmp_path):
    from seeding import rng_for_key
    captured = _capture(monkeypatch)
    df = _test_df()
    common.create_hypothesis_scatter_plots(
        "H1", HYPO, df, "Inhibit_t", "interference", "stroop_effect",
        tmp_path, level="channel", seed=20260922)
    expected = common.test_correlation(
        df["Inhibit_t"].values, df["activation"].values, "H1_self_brain",
        rng=rng_for_key("H1_self_brain", 20260922))
    assert captured["H1_self_brain"]["p"] == expected["p"]
    assert captured["H1_self_brain"]["ci_lower"] == expected["ci_lower"]
