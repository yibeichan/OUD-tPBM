"""Readers for the per-session files ``step2c_session_info.py`` writes.

Both readers resolve ``OUTPUT_DIR`` from the environment at call time (not
at import time) when ``output_dir`` is not given, so tests that monkeypatch
``OUTPUT_DIR`` see the change without a reload. Each resolved directory's
CSVs are read once and cached.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_short_channels_cache: dict[tuple[str, str], pd.DataFrame] = {}
_condition_counts_cache: dict[str, pd.DataFrame] = {}


def _session_info_dir(output_dir: str | None) -> Path:
    if output_dir is None:
        output_dir = os.environ["OUTPUT_DIR"]
    return Path(output_dir) / "04b_session_info"


def _load_short_channels(task: str, output_dir: str | None) -> pd.DataFrame:
    session_dir = _session_info_dir(output_dir)
    key = (str(session_dir), task)
    if key not in _short_channels_cache:
        path = session_dir / f"short_channels_{task}.csv"
        if not path.exists():
            raise FileNotFoundError(f"short-channel file not found: {path}")
        _short_channels_cache[key] = pd.read_csv(path)
    return _short_channels_cache[key]


def _load_condition_counts(output_dir: str | None) -> pd.DataFrame:
    session_dir = _session_info_dir(output_dir)
    key = str(session_dir)
    if key not in _condition_counts_cache:
        path = session_dir / "wcst_condition_counts.csv"
        if not path.exists():
            raise FileNotFoundError(f"WCST condition-count file not found: {path}")
        _condition_counts_cache[key] = pd.read_csv(path)
    return _condition_counts_cache[key]


def short_channels(task: str, session_id: str, output_dir: str | None = None) -> frozenset[str]:
    """Return the canonical (S#_D#) short-channel names for one session."""
    df = _load_short_channels(task, output_dir)
    session_df = df[df.session_id == session_id]
    if session_df.empty:
        raise KeyError(f"session {session_id!r} not found in short_channels_{task}.csv")
    return frozenset(session_df.loc[session_df.is_short, "channel"])


def sessions_without_regular_incorrect(output_dir: str | None = None) -> frozenset[str]:
    """Return the WCST session_ids whose regular_incorrect count is zero."""
    df = _load_condition_counts(output_dir)
    return frozenset(df.loc[df.regular_incorrect == 0, "session_id"])


def session_lacks_regular_incorrect(session_id: str, output_dir: str | None = None) -> bool:
    """Return whether one WCST session has zero regular_incorrect trials.

    Raises KeyError if ``session_id`` is absent from wcst_condition_counts.csv,
    so a session missing from that file is never silently treated as having
    regular-incorrect trials (i.e. left unmasked).
    """
    df = _load_condition_counts(output_dir)
    session_df = df[df.session_id == session_id]
    if session_df.empty:
        raise KeyError(
            f"session {session_id!r} not found in wcst_condition_counts.csv"
        )
    return bool((session_df["regular_incorrect"] == 0).all())
