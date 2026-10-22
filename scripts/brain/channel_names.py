"""Canonical fNIRS channel names, stable across sessions.

MNE's SNIRF reader names a channel by the 1-based indices of its source and
detector in that recording's probe lists ("S23_D92"). Every recording in
this study has the same full Kernel Flow probe except S4's, which lacks one
source and three detectors, so from source index 31 and detector index 61
its index names point at the next optode along. Channel-level analyses pool
channels by name across participants, so those names must mean the same
optode everywhere.

The canonical index of an optode is its 1-based position in the full
probe's label order (config/flow_probe_template.json). For a full-probe
recording the map is the identity; for S4 it translates through the probe
labels, which sit at identical positions in every recording.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import mne
import numpy as np

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config" / "flow_probe_template.json"
_NAME_RE = re.compile(r"S(\d+)_D(\d+)")


@lru_cache(maxsize=4)
def load_template(path: Path = TEMPLATE_PATH) -> tuple[tuple[str, ...], tuple[str, ...]]:
    data = json.loads(Path(path).read_text())
    return tuple(data["source_labels"]), tuple(data["detector_labels"])


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_probe_labels(snirf_path) -> tuple[list[str], list[str]]:
    with h5py.File(snirf_path, "r") as h5:
        nirs = h5[sorted(k for k in h5.keys() if k.startswith("nirs"))[0]]
        probe = nirs["probe"]
        return ([_decode(v) for v in probe["sourceLabels"][()]],
                [_decode(v) for v in probe["detectorLabels"][()]])


def index_maps(source_labels, detector_labels, template=None
               ) -> tuple[dict[int, int], dict[int, int]]:
    """Map 1-based native optode indices to 1-based template indices."""
    for kind, labels in (("source", source_labels), ("detector", detector_labels)):
        dupes = sorted(l for l, n in Counter(labels).items() if n > 1)
        if dupes:
            raise ValueError(f"recording has duplicate {kind} labels: {dupes[:10]}")
    t_src, t_det = template or load_template()
    src_pos = {label: i + 1 for i, label in enumerate(t_src)}
    det_pos = {label: i + 1 for i, label in enumerate(t_det)}
    unknown = [l for l in source_labels if l not in src_pos] + \
              [l for l in detector_labels if l not in det_pos]
    if unknown:
        raise ValueError(f"probe labels not in the template: {unknown[:10]}")
    return ({i + 1: src_pos[l] for i, l in enumerate(source_labels)},
            {i + 1: det_pos[l] for i, l in enumerate(detector_labels)})


def canonical_maps_for_snirf(snirf_path) -> tuple[dict[int, int], dict[int, int]]:
    return index_maps(*read_probe_labels(snirf_path))


def canonical_name(name: str, src_map: dict[int, int], det_map: dict[int, int]) -> str:
    base, sep, suffix = name.partition(" ")
    match = _NAME_RE.fullmatch(base)
    if match is None:
        raise ValueError(f"not an S#_D# channel name: {name!r}")
    new = f"S{src_map[int(match.group(1))]}_D{det_map[int(match.group(2))]}"
    return new + sep + suffix


def rename_raw_to_canonical(raw, snirf_path) -> int:
    """Rename a raw recording's S#_D# channels to canonical names, in place."""
    src_map, det_map = canonical_maps_for_snirf(snirf_path)
    changed = {}
    for ch in raw.ch_names:
        if _NAME_RE.match(ch):
            new = canonical_name(ch, src_map, det_map)
            if new != ch:
                changed[ch] = new
    if changed:
        raw.rename_channels(changed)
    return len(changed)


def build_canonical_montage(snirf_paths) -> tuple["mne.Info", dict[str, int]]:
    """HbO montage covering every canonical channel of the given recordings.

    No single recording has every channel, so the montage is their union.
    After canonical renaming a name has one position in every recording;
    this raises if that ever fails, or if two names share a position.
    """
    chs: dict[str, dict] = {}
    sfreq = None
    for path in sorted(str(p) for p in snirf_paths):
        raw = mne.io.read_raw_snirf(path, preload=False, verbose="error")
        rename_raw_to_canonical(raw, path)
        sfreq = sfreq or raw.info["sfreq"]
        for ch in raw.info["chs"]:
            if not ch["ch_name"].endswith("hbo"):
                continue
            prev = chs.get(ch["ch_name"])
            if prev is not None and np.nanmax(np.abs(prev["loc"][:9] - ch["loc"][:9])) > 1e-6:
                raise ValueError(f"{ch['ch_name']} has two positions ({path})")
            chs.setdefault(ch["ch_name"], ch)
    seen: dict[tuple, str] = {}
    for name, ch in chs.items():
        key = tuple(np.round(ch["loc"][:3], 6))
        if key in seen:
            raise ValueError(f"{name} and {seen[key]} share a position")
        seen[key] = name
    names = sorted(chs)
    info = mne.create_info(names, sfreq, ch_types="hbo", verbose=False)
    with info._unlock():
        for i, name in enumerate(names):
            info["chs"][i]["loc"] = chs[name]["loc"].copy()
            info["chs"][i]["coord_frame"] = chs[name]["coord_frame"]
    return info, {name.split(" ")[0]: i for i, name in enumerate(names)}
