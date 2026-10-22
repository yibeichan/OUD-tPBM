"""step2a must key mappings canonically but resolve parcels through the
session's own probe labels."""
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(BRAIN))
import channel_names as cn  # noqa: E402


TEMPLATE = (("M000S00", "M001S00", "M002S00"),
            ("M000D00", "M001D00", "M002D00"))


def _snirf(path, src, det, pairs):
    with h5py.File(path, "w") as h:
        p = h.create_group("nirs/probe")
        p["sourceLabels"] = np.array([s.encode() for s in src])
        p["detectorLabels"] = np.array([d.encode() for d in det])
        data = h.create_group("nirs/data1")
        for i, (s, d) in enumerate(pairs, start=1):
            ml = data.create_group(f"measurementList{i}")
            ml["sourceIndex"] = s
            ml["detectorIndex"] = d


def test_lookup_keys_canonical_parcels_native(tmp_path, monkeypatch):
    # step2a exits at import without these; a clean checkout has no .env. load_dotenv does
    # not override, so a local .env cannot redirect this test either.
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    spec = importlib.util.spec_from_file_location("step2a", BRAIN / "step2a_glm_to_parcellation.py")
    step2a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step2a)
    monkeypatch.setattr(cn, "load_template", lambda path=None: TEMPLATE)

    atlas_dir = tmp_path / "data" / "atlas"
    atlas_dir.mkdir(parents=True)
    np.savez(atlas_dir / "flow2_absorption_Schaefer2018_100Parcels_7Networks.npz",
             source_labels=np.array(["m000s00", "m002s00"]),
             detector_labels=np.array(["m000d00", "m002d00"]))
    monkeypatch.setattr(step2a, "BASE_DIR", str(tmp_path))

    # This recording lacks M001S00 and M001D00: native index 2 is optode M002.
    snirf = tmp_path / "rec.snirf"
    _snirf(snirf, ["M000S00", "M002S00"], ["M000D00", "M002D00"], [(1, 1), (2, 2)])
    lookup = step2a.ChannelMapper(str(snirf)).create_glm_to_parcellation_lookup()

    assert lookup == {"S1_D1": 0, "S3_D3": 1}
