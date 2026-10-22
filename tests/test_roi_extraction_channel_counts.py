"""step3b's extraction stats count channels, not contrast rows.

The GLM frame has one row per channel per contrast; ``total_channels`` and
``mapped_channels`` count each channel once, not once per contrast. No study
data is read.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

_BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"


@pytest.fixture(scope="module")
def step3b(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_BRAIN))
        module = importlib.import_module("step3b_roi_extraction")
        yield module
    for name in ("step3b_roi_extraction", "roi_utils"):
        sys.modules.pop(name, None)


def test_stats_count_each_hbo_channel_once(step3b, monkeypatch):
    contrasts = ["stroop_effect", "mixing_cost", "incongruent_vs_pure"]
    rows = []
    for ch in ["S1_D1", "S2_D2", "S3_D3"]:
        for chrom in ("hbo", "hbr"):
            for c in contrasts:
                rows.append({"channel": ch, "chromophore": chrom, "contrast": c,
                             "effect": 0.1, "stat": 1.0, "p_value": 0.3})
    glm = pd.DataFrame(rows)
    mapping = {"channel_to_network": {"S1_D1": "Cont", "S2_D2": "Cont"},
               "channel_to_network_roi": {"S1_D1": "Cont_PFCl", "S2_D2": "Cont_PFCl"},
               "channel_to_full_roi": {}, "channel_to_parcel_idx": {}}
    monkeypatch.setattr(step3b, "load_glm_results", lambda f, task, sid: glm)
    monkeypatch.setattr(step3b, "load_roi_mapping", lambda f: mapping)

    _, stats = step3b.process_single_subject(
        Path("S1_aaaaaaa_glm_contrasts.csv"), Path("mapping.json"), "stroop", contrasts)

    assert stats["total_channels"] == 3
    assert stats["mapped_channels"] == 2
    assert stats["mapping_rate"] == pytest.approx(2 / 3)
