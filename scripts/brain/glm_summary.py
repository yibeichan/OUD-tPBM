"""Per-session GLM run summaries and the shared analysis_summary.json.

The first-level GLM runs as a SLURM array, one session per task, and tasks
can finish together. So no task edits a shared file in place: each session
writes its own file under ``summaries/`` (a path no other task touches), and
``analysis_summary.json`` is rebuilt from those files under a lock and
replaced atomically. The list holds one entry per session (a re-run replaces
its session's entry), ordered by file name.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMARY_NAME = "analysis_summary.json"
SESSION_DIR = "summaries"


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def write_session_summary(log_dir, summary: dict) -> Path | None:
    """Write this session's summary, then rebuild the shared summary.

    ``log_dir`` is ``OUTPUT_DIR/logs/GLM/<task>``. This runs from a ``finally``
    block, so it never raises: a failure (lock error on NFS, unreadable or
    corrupt per-session file) is logged as a warning and the shared file is
    left as it was. The per-session files are the record. Returns the shared
    file, or None when it could not be rebuilt.
    """
    log_dir = Path(log_dir)
    session_dir = log_dir / SESSION_DIR
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        name = f"{summary['subject_id']}_{summary['task_id']}_summary.json"
        _atomic_write_json(session_dir / name, summary)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not write the session summary under %s: %s", session_dir, exc)
        return None

    try:
        with open(log_dir / ".analysis_summary.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = []
            for path in sorted(session_dir.glob("*_summary.json")):
                with open(path) as f:
                    entries.append(json.load(f))
            shared = log_dir / SUMMARY_NAME
            _atomic_write_json(shared, entries)
        return shared
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not rebuild %s (%s); the per-session files in %s are the record",
            log_dir / SUMMARY_NAME, exc, session_dir)
        return None
