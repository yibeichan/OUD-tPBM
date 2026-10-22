"""Trial onsets on the recording's own clock.

The event table's ``timestamp`` and the SNIRF data axis are different clocks.
The SNIRF export is cropped to the experiment (about 1 s of margin at each
end) and re-zeroed, and its ``/nirs/stim*`` onsets are re-based with it; the
event table is not. Within a session the two differ by one constant offset,
a whole number of samples, that varies across sessions. Placing a trial
at ``timestamp * sfreq`` therefore lags every regressor by that offset.

So the onsets come from the ``StartTrial`` stims MNE reads from the SNIRF
file, matched in order to the event table's ``start_trial`` rows. The event
table still supplies each trial's labels and duration. The match is checked,
not assumed: the counts must agree, and the table-minus-SNIRF offset must be
constant to one sample.
"""
from __future__ import annotations

import numpy as np

SNIRF_TRIAL_DESCRIPTION = "StartTrial"


def snirf_stim_onsets(raw, description: str = SNIRF_TRIAL_DESCRIPTION) -> np.ndarray:
    """Return the onsets (s, on ``raw.times``) of the SNIRF stims named ``description``.

    Read from ``raw.annotations`` as MNE loaded them from the file, so call
    this before anything replaces the annotations. With an ``orig_time`` the
    onsets count from ``meas_date``, and the first sample sits at
    ``raw.first_time``; without one they count from the first sample.
    """
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        raise ValueError("the recording carries no annotations; cannot read SNIRF stim onsets")
    mask = np.asarray(annotations.description) == description
    onsets = np.asarray(annotations.onset, dtype=float)[mask]
    if annotations.orig_time is not None:
        onsets = onsets - float(getattr(raw, "first_time", 0.0))
    return np.sort(onsets)


def match_trial_onsets(table_timestamps, snirf_onsets, sfreq: float,
                       n_times: int | None = None) -> tuple[np.ndarray, float]:
    """Pair each event-table trial with its SNIRF onset, in order.

    Parameters
    ----------
    table_timestamps
        The event table's ``start_trial`` timestamps, ascending.
    snirf_onsets
        The SNIRF ``StartTrial`` onsets on the data axis, ascending.
    sfreq
        Sampling rate (Hz); one sample is the tolerance on the offset.
    n_times
        Samples in the recording. When given, every onset must fall inside it.

    Returns
    -------
    onsets : np.ndarray
        The SNIRF onset for each table row, in the table's order.
    offset : float
        The session's table-minus-SNIRF offset in seconds (median over trials).

    Raises
    ------
    ValueError
        If the counts differ, there are no trials, the table is not sorted, the
        offset varies by more than one sample, or an onset lies outside the
        recording.
    """
    table = np.asarray(table_timestamps, dtype=float)
    snirf = np.asarray(snirf_onsets, dtype=float)
    if len(table) != len(snirf):
        raise ValueError(
            f"event table has {len(table)} start_trial rows but the SNIRF file has "
            f"{len(snirf)} {SNIRF_TRIAL_DESCRIPTION} stims; cannot pair trials to onsets"
        )
    if len(table) == 0:
        raise ValueError("no trials in the event table")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(snirf))):
        raise ValueError("trial timestamps and SNIRF onsets must all be finite")
    if np.any(np.diff(table) < 0) or np.any(np.diff(snirf) < 0):
        raise ValueError("trial timestamps and SNIRF onsets must both be ascending")

    offsets = table - snirf
    spread = float(offsets.max() - offsets.min())
    if spread > 1.0 / sfreq:
        raise ValueError(
            f"event-table minus SNIRF offset is not constant: it ranges "
            f"{offsets.min():.3f}-{offsets.max():.3f} s ({spread:.3f} s, more than one "
            f"sample at {sfreq:.3f} Hz), so the trials cannot be paired in order"
        )

    if n_times is not None:
        samples = np.floor(snirf * sfreq).astype(int)
        outside = np.flatnonzero((samples < 0) | (samples >= n_times))
        if outside.size:
            raise ValueError(
                f"{outside.size} SNIRF trial onset(s) fall outside the recording "
                f"(0-{n_times} samples), at trial positions {outside[:10].tolist()}"
            )

    return snirf, float(np.median(offsets))
