#!/usr/bin/env python3
"""Check canonical channel names over every SNIRF under BASE_DIR/data.

Asserts that full-probe recordings map to the identity, reports how many of
each recording's HbO channels are renamed, and asserts that after renaming
one name never has two positions and two names never share a position.
"""
import glob
import os
import sys
from pathlib import Path

import mne
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_names import canonical_maps_for_snirf, rename_raw_to_canonical  # noqa: E402

load_dotenv()
DATA = Path(os.environ["BASE_DIR"]) / "data"
failures = 0
for task in ("stroop", "wcst"):
    positions: dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(str(DATA / task / "*.snirf"))):
        src_map, det_map = canonical_maps_for_snirf(f)
        identity = all(k == v for k, v in src_map.items()) and all(k == v for k, v in det_map.items())
        raw = mne.io.read_raw_snirf(f, preload=False, verbose="error")
        full = len(src_map) == 120 and len(det_map) == 240
        if full and not identity:
            print(f"FAIL {task} {Path(f).name}: full probe but map is not the identity"); failures += 1
        n = rename_raw_to_canonical(raw, f)
        hbo = [c for c in raw.info["chs"] if c["ch_name"].endswith("hbo")]
        print(f"{task} {Path(f).name}: sources={len(src_map)} detectors={len(det_map)} "
              f"renamed={n // 2 if n else 0} HbO channels")
        for ch in hbo:
            name, loc = ch["ch_name"].split(" ")[0], ch["loc"][:9]
            if name in positions and np.nanmax(np.abs(positions[name] - loc)) > 1e-6:
                print(f"FAIL {task} {name}: two positions"); failures += 1
            positions.setdefault(name, loc.copy())
    mids = {}
    for name, loc in positions.items():
        key = tuple(np.round(loc[:3], 6))
        if key in mids:
            print(f"FAIL {task}: {name} and {mids[key]} share a position"); failures += 1
        mids[key] = name
    print(f"{task}: {len(positions)} canonical HbO channels")
sys.exit(1 if failures else 0)
