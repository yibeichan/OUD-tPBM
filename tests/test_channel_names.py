"""Tests for canonical channel names (scripts/brain/channel_names.py).

A recording whose probe lacks optodes gets index-based MNE names that point
at different optodes than in a full-probe recording. Canonical names number
each optode by its position in the full probe's label order instead.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "brain"))
import channel_names as cn  # noqa: E402


TEMPLATE = (("M000S00", "M001S00", "M002S00"),
            ("M000D00", "M000D01", "M001D00", "M002D00"))


def test_full_probe_is_identity():
    src, det = cn.index_maps(list(TEMPLATE[0]), list(TEMPLATE[1]), template=TEMPLATE)
    assert src == {1: 1, 2: 2, 3: 3}
    assert det == {1: 1, 2: 2, 3: 3, 4: 4}


def test_missing_optodes_shift_to_template_indices():
    # M001S00 and M000D01 absent: native indices after the gap move up.
    src, det = cn.index_maps(["M000S00", "M002S00"], ["M000D00", "M001D00", "M002D00"],
                             template=TEMPLATE)
    assert src == {1: 1, 2: 3}
    assert det == {1: 1, 2: 3, 3: 4}


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="M099S00"):
        cn.index_maps(["M000S00", "M099S00"], list(TEMPLATE[1]), template=TEMPLATE)


def test_duplicate_recording_label_raises():
    # Two native indices on one label would map to one canonical index.
    with pytest.raises(ValueError, match="duplicate source labels: \\['M001S00'\\]"):
        cn.index_maps(["M000S00", "M001S00", "M001S00"], list(TEMPLATE[1]), template=TEMPLATE)
    with pytest.raises(ValueError, match="duplicate detector labels: \\['M000D00'\\]"):
        cn.index_maps(list(TEMPLATE[0]), ["M000D00", "M000D00"], template=TEMPLATE)


def test_canonical_name_keeps_suffix():
    src, det = {2: 3}, {2: 3}
    assert cn.canonical_name("S2_D2 hbo", src, det) == "S3_D3 hbo"
    assert cn.canonical_name("S2_D2", src, det) == "S3_D3"


def test_canonical_name_rejects_other_formats():
    with pytest.raises(ValueError):
        cn.canonical_name("M000S00_M000D00", {1: 1}, {1: 1})


def test_rename_keeps_hbo_hbr_pairs(monkeypatch):
    class FakeRaw:
        def __init__(self):
            self.ch_names = ["S2_D2 hbo", "S2_D2 hbr", "S1_D1 hbo", "S1_D1 hbr"]
        def rename_channels(self, mapping):
            self.ch_names = [mapping.get(c, c) for c in self.ch_names]
    monkeypatch.setattr(cn, "canonical_maps_for_snirf",
                        lambda path: ({1: 1, 2: 3}, {1: 1, 2: 3}))
    raw = FakeRaw()
    assert cn.rename_raw_to_canonical(raw, "unused.snirf") == 2
    assert raw.ch_names == ["S3_D3 hbo", "S3_D3 hbr", "S1_D1 hbo", "S1_D1 hbr"]


def test_committed_template_is_the_full_probe():
    src, det = cn.load_template()
    assert len(src) == 120 and len(det) == 240
    assert len(set(src)) == 120 and len(set(det)) == 240


def test_union_montage_has_no_collisions(monkeypatch):
    import numpy as np

    def fake_read(path, preload=False, verbose=None):
        class Raw:
            pass
        raw = Raw()
        names = {"a": ["S1_D1 hbo", "S1_D1 hbr"], "b": ["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo", "S2_D2 hbr"]}[path]
        loc = {"S1_D1": [0.0, 0, 0], "S2_D2": [1.0, 0, 0]}
        raw.ch_names = names
        raw.info = {"sfreq": 10.0, "chs": [
            {"ch_name": n, "loc": np.array(loc[n.split(" ")[0]] + [0] * 9, float), "coord_frame": 4}
            for n in names]}
        raw.rename_channels = lambda m: None
        return raw

    monkeypatch.setattr(cn.mne.io, "read_raw_snirf", fake_read)
    monkeypatch.setattr(cn, "rename_raw_to_canonical", lambda raw, path: 0)
    info, mapping = cn.build_canonical_montage(["a", "b"])
    assert sorted(mapping) == ["S1_D1", "S2_D2"]


def _fake_reader(recordings):
    """read_raw_snirf stand-in: recordings maps path -> {name: xyz}."""
    import numpy as np

    def fake_read(path, preload=False, verbose=None):
        class Raw:
            pass
        raw = Raw()
        raw.info = {"sfreq": 10.0, "chs": [
            {"ch_name": f"{name} {kind}",
             "loc": np.array(list(xyz) + [0] * 9, float), "coord_frame": 4}
            for name, xyz in recordings[path].items() for kind in ("hbo", "hbr")]}
        raw.ch_names = [ch["ch_name"] for ch in raw.info["chs"]]
        return raw
    return fake_read


def test_union_montage_raises_on_one_name_two_positions(monkeypatch):
    monkeypatch.setattr(cn.mne.io, "read_raw_snirf", _fake_reader({
        "a": {"S1_D1": (0.0, 0.0, 0.0)},
        "b": {"S1_D1": (0.01, 0.0, 0.0)},
    }))
    monkeypatch.setattr(cn, "rename_raw_to_canonical", lambda raw, path: 0)
    with pytest.raises(ValueError, match="S1_D1 hbo has two positions"):
        cn.build_canonical_montage(["a", "b"])


def test_union_montage_raises_on_two_names_one_position(monkeypatch):
    monkeypatch.setattr(cn.mne.io, "read_raw_snirf", _fake_reader({
        "a": {"S1_D1": (0.0, 0.0, 0.0)},
        "b": {"S2_D2": (0.0, 0.0, 0.0)},
    }))
    monkeypatch.setattr(cn, "rename_raw_to_canonical", lambda raw, path: 0)
    with pytest.raises(ValueError, match="share a position"):
        cn.build_canonical_montage(["a", "b"])
