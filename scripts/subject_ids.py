"""Canonical subject identifiers.

A subject is identified by an S-code: the letter ``S`` followed by a
positive integer with no zero padding (``S1`` … ``S12``).  One form is used
everywhere -- input filenames, dataframe columns, output tables and figure
labels.

Two properties are stated here because a silent failure of either would
corrupt results without an error:

* **Parsing is anchored.**  Identifiers are matched against the whole
  filename rather than read from a fixed ``split('_')`` offset, so adding
  or removing a filename component raises instead of returning a different
  token.
* **Ordering is numeric.**  ``sorted()`` on unpadded codes places ``S10``
  before ``S2``.  Anything that orders subjects for a table, a figure axis
  or a groupby must use :func:`subject_sort_key`.

Nothing here falls back to returning its input.  A filename that carries no
identifier raises, because such a value's only remaining use is to be
interpolated into an output path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

#: A canonical subject code: ``S`` followed by an unpadded positive integer.
SUBJECT_RE = re.compile(r"^S([1-9][0-9]*)$")

_SUBJECT = r"S[1-9][0-9]*"
_SESSION = r"[0-9a-f]{7}"

# Raw acquisition files:  NeuroLume_S1_000000a_5.snirf
#                         NeuroLume_S1_000000a_task_events.tsv
_RAW_RE = re.compile(rf"^NeuroLume_(?P<subject>{_SUBJECT})_(?P<session>{_SESSION})_")

# Derived per-session files:  S1_000000a_glm_contrasts.csv
#                             S1_000000a_results.json
_DERIVED_RE = re.compile(rf"^(?P<subject>{_SUBJECT})_(?P<session>{_SESSION})_")

# Session-scoped mappings:
#   mapping_stroop_01_000000a.json / roi_mapping_wcst_12_0000014.json
#
# The middle component is the task name followed by a subject component. These
# names therefore do carry a subject, but this module deliberately does not
# parse one out of them: extract_session_id is the only accessor they need, and
# no caller asks them for a subject. The component is left unconstrained
# because the task vocabulary is not this module's business to enumerate.
#
# The greedy ``.+`` cannot swallow the session: ``session`` is anchored to ``$``
# and fixed at exactly seven hex characters, so the only viable split is at the
# last underscore, and greediness makes that the one chosen. Aggregate mappings
# that are not session-scoped (``channel_roi_mapping.json``,
# ``glm_to_parcellation_mapping.json``) do not match and therefore raise.
_MAPPING_RE = re.compile(rf"^(?:roi_)?mapping_.+_(?P<session>{_SESSION})$")

# Tolerated input spellings on the way in: s1, S01, S-1.
_LOOSE_RE = re.compile(r"^s[-_]?0*([1-9][0-9]*)$", re.IGNORECASE)


def task_events_filename(subject: str, task_id: str) -> str:
    """Return the raw event-table filename of one acquisition session."""
    return f"NeuroLume_{subject}_{task_id}_task_events.tsv"


def is_subject_id(value: str) -> bool:
    """Return whether ``value`` is already a canonical subject code."""
    return isinstance(value, str) and SUBJECT_RE.match(value) is not None


def subject_sort_key(subject: str) -> int:
    """Return the integer to sort ``subject`` by.

    Use as ``key=subject_sort_key`` wherever subjects are ordered.
    """
    match = SUBJECT_RE.match(subject)
    if match is None:
        raise ValueError(f"not a subject identifier: {subject!r}")
    return int(match.group(1))


def sort_subjects(subjects: Iterable[str]) -> list[str]:
    """Return ``subjects`` in numeric order."""
    return sorted(subjects, key=subject_sort_key)


def standardize_subject_id(value: str) -> str:
    """Return ``value`` as a canonical subject code.

    Accepts ``S1``, ``s1``, ``S01`` and ``S-1``; raises on anything else.
    """
    if not isinstance(value, str):
        raise ValueError(f"not a subject identifier: {value!r}")
    candidate = value.strip()
    if SUBJECT_RE.match(candidate):
        return candidate
    match = _LOOSE_RE.match(candidate)
    if match is None:
        raise ValueError(f"not a subject identifier: {value!r}")
    return f"S{int(match.group(1))}"


def extract_subject_id(filename: str) -> str:
    """Return the subject code carried by ``filename``.

    The session-scoped mapping families (``mapping_*``, ``roi_mapping_*``) are
    not supported here and always raise; use :func:`extract_session_id` for
    those.

    Args:
        filename: A bare filename, not a path.

    Raises:
        ValueError: if the name carries no subject field.
    """
    stem = Path(filename).name
    for pattern in (_RAW_RE, _DERIVED_RE):
        match = pattern.match(stem)
        if match:
            return match.group("subject")
    raise ValueError(f"no subject identifier in filename: {filename!r}")


def extract_session_id(filename: str) -> str:
    """Return the 7-character session ID carried by ``filename``.

    In addition to the raw and derived families, this accepts the
    session-scoped mapping families (``mapping_*``, ``roi_mapping_*``) that
    :func:`extract_subject_id` rejects.

    Args:
        filename: A bare filename, not a path.

    Raises:
        ValueError: if the name carries no session field.
    """
    stem = Path(filename).name
    for pattern in (_RAW_RE, _DERIVED_RE):
        match = pattern.match(stem)
        if match:
            return match.group("session")
    match = _MAPPING_RE.match(Path(stem).stem)
    if match:
        return match.group("session")
    raise ValueError(f"no session identifier in filename: {filename!r}")


def subject_column(columns: Iterable[str]) -> str:
    """Return the one canonical subject code among ``columns``.

    Per-subject assessment frames hold item metadata columns plus one column
    named by the subject's code. Selecting it by name rather than by position
    means an added metadata column cannot silently become the one scored.

    Raises:
        ValueError: unless exactly one column is a canonical subject code.
    """
    found = [c for c in columns if is_subject_id(c)]
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one subject column, found {found} in {list(columns)}"
        )
    return found[0]
