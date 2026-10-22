import json
from pathlib import Path

import pytest

from scripts.analysis_variant import (
    AnalysisVariant, VariantError, VARIANT_FILENAME, load_variant,
)

MAIN = {
    "name": "main", "channels": "long", "exclude_low_accuracy": True,
    "h2_metric": "perseverative_error_rate", "exclude_invalid_brief": True,
    "second_timepoint": "most_complete",
    "mask_wcst_without_regular_incorrect": True, "brain_direction": "efficiency",
}


def _write(tmp_path, payload):
    (tmp_path / VARIANT_FILENAME).write_text(json.dumps(payload))
    return tmp_path


def test_loads_every_field(tmp_path):
    v = load_variant(_write(tmp_path, MAIN))
    assert v == AnalysisVariant(**MAIN)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(VariantError, match=VARIANT_FILENAME):
        load_variant(tmp_path)


def test_unknown_key_is_named(tmp_path):
    bad = dict(MAIN); bad["channel"] = bad.pop("channels")
    with pytest.raises(VariantError, match="channel"):
        load_variant(_write(tmp_path, bad))


def test_missing_key_is_named(tmp_path):
    bad = dict(MAIN); del bad["brain_direction"]
    with pytest.raises(VariantError, match="brain_direction"):
        load_variant(_write(tmp_path, bad))


@pytest.mark.parametrize("key,value", [
    ("channels", "short"), ("h2_metric", "rate"), ("second_timepoint", "post"),
    ("brain_direction", "up"), ("exclude_low_accuracy", "yes"),
])
def test_bad_value_is_named(tmp_path, key, value):
    bad = dict(MAIN); bad[key] = value
    with pytest.raises(VariantError, match=key):
        load_variant(_write(tmp_path, bad))


def test_committed_variant_files_all_load():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "config" / "analysis_variants"
    names = sorted(p.stem for p in root.glob("*.json"))
    assert names == ["main", "supp_all_channels",
                     "supp_brain_activation", "supp_count", "supp_low_accuracy"]
    for p in root.glob("*.json"):
        v = load_variant(p.parent / "__unused__", path=p)
        assert v.name == p.stem


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "analysis_variants"


def test_legacy_second_timepoint_rejected(tmp_path):
    cfg = json.loads((CONFIG_DIR / "main.json").read_text())
    cfg["name"] = "legacy_probe"
    cfg["second_timepoint"] = "legacy"
    p = tmp_path / "legacy_probe.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(VariantError, match="second_timepoint"):
        load_variant(path=p)
