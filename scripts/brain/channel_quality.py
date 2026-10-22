"""Drop channels that carry no signal before the first-level GLM.

A channel whose series is constant (a dead optode pair) or contains a
non-finite value has no estimable beta. Left in, it would be written out as
effect 0, se 0, p 0.5, which later stages would read as a measured zero, and
its singular AR(1) fit would push every channel of the session to OLS. So it
is dropped before ``run_glm``: it is absent from the session's outputs
(missing downstream, not zero), and the drop is recorded.

Both chromophores of a source-detector pair are dropped together: MNE's
fNIRS helpers expect hbo/hbr pairs, and on the study data a dead pair is
flat in both.
"""
from __future__ import annotations

import numpy as np


def _pair(ch_name: str) -> str:
    return ch_name.split(" ")[0]


def unusable_channels(raw) -> list[str]:
    """Return the channels to drop: zero-variance or non-finite, plus their pair partners."""
    data = np.asarray(raw.get_data(), dtype=float)
    finite = np.isfinite(data).all(axis=1)
    flat = np.zeros(len(raw.ch_names), dtype=bool)
    flat[finite] = np.ptp(data[finite], axis=1) == 0
    bad_pairs = {_pair(name) for name, bad in zip(raw.ch_names, flat | ~finite) if bad}
    return [name for name in raw.ch_names if _pair(name) in bad_pairs]


def drop_unusable_channels(raw) -> list[str]:
    """Drop :func:`unusable_channels` from ``raw`` in place and return their names."""
    dropped = unusable_channels(raw)
    if dropped:
        raw.drop_channels(dropped)
    return dropped
