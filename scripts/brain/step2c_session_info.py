#!/usr/bin/env python3
"""
Per-session short channels and WCST condition counts (Step 04b).

For every Stroop and WCST session this records, per HbO channel, the
source-detector distance and whether it is "short" (<10 mm) by the same
strict threshold ``mne_nirs.channels.get_short_channels`` uses for step1's
nuisance regressor. For WCST it additionally records how many trials of
each condition (regular_correct, regular_incorrect, change, post_change)
``step1_glm_wcst``'s event processing produces for that session.

Later stages use these to drop short channels from channel-level analyses
and to mask WCST contrasts built on a condition that a session has zero
trials of.

Outputs (under ``OUTPUT_DIR/04b_session_info/``):
    short_channels_stroop.csv, short_channels_wcst.csv
        subject, session_id, task_order, channel, distance_mm, is_short
    wcst_condition_counts.csv
        subject, session_id, task_order, change, post_change,
        regular_correct, regular_incorrect
"""
import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

import mne
import pandas as pd
from dotenv import load_dotenv
from mne.preprocessing.nirs import source_detector_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_names import rename_raw_to_canonical  # noqa: E402
import step1_glm_wcst  # noqa: E402
from step1_glm_wcst import read_event_table  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402
from subject_ids import task_events_filename  # noqa: E402
from session_exclusions import write_exclusion_log  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Short-channel threshold in metres. Must equal the ``max_dist`` default of
# mne_nirs.channels.get_short_channels, which step1 uses for its nuisance
# regressors (tests/test_step2c_short_channel_threshold.py checks this).
SHORT_CHANNEL_MAX_DIST_M = 0.01

_WCST_CONDITIONS = ("change", "post_change", "regular_correct", "regular_incorrect")


def refuse_unsafe_output_dir(output_dir: str) -> None:
    """Raise if ``output_dir`` names a protected run tree (see analysis_variant.protected_tree_markers)."""
    try:
        refuse_protected_tree(output_dir)
    except RuntimeError as exc:
        raise RuntimeError(
            f"refusing to write session info into {output_dir!r}: {exc}"
        ) from None


@contextlib.contextmanager
def _logging_silenced():
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def _wcst_analyzer_class() -> type:
    """Return step1's WCST analyzer class (identified by ``process_events``)."""
    return next(
        value for value in vars(step1_glm_wcst).values()
        if isinstance(value, type) and hasattr(value, "process_events")
    )


def hbo_short_channel_rows(raw, subject: str, session_id: str, task_order: int) -> list[dict]:
    """Return one row per HbO channel: canonical name, distance, is_short."""
    hbo = [i for i, name in enumerate(raw.ch_names) if name.endswith("hbo")]
    distances = source_detector_distances(raw.info, picks=hbo)
    rows = []
    for i, dist in zip(hbo, distances):
        dist = float(dist)
        rows.append({
            "subject": subject,
            "session_id": session_id,
            "task_order": int(task_order),
            "channel": raw.ch_names[i].split(" ")[0],
            "distance_mm": round(dist * 1000, 3),
            "is_short": dist < SHORT_CHANNEL_MAX_DIST_M,
        })
    return rows


def wcst_condition_count_row(base_dir: str, subject: str, task_id: str, task_order: int,
                              raw, analyzer_cls: type) -> dict:
    """Return one row of WCST condition trial counts for a single session.

    ``raw`` is the session's recording (data need not be loaded): step1 pairs
    the event table's trials with the SNIRF ``StartTrial`` stims, so the
    counts come from exactly the event processing step1 fits.
    """
    events_file = os.path.join(
        base_dir, "data", "wcst", task_events_filename(subject, task_id)
    )
    analyzer = analyzer_cls("wcst")
    analyzer.events_df = read_event_table(events_file)
    analyzer.raw = raw
    with _logging_silenced():
        events, condition_mapping, _durations = analyzer.process_events()
    counts = pd.Series(events[:, 2]).map(condition_mapping).value_counts().to_dict()
    row = {"subject": subject, "session_id": task_id, "task_order": int(task_order)}
    row.update({name: int(counts.get(name, 0)) for name in _WCST_CONDITIONS})
    return row


def build_task_session_info(base_dir: str, task: str, analyzer_cls: type
                             ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (short-channel rows, WCST condition-count rows or None) for a task."""
    task_info = pd.read_csv(Path(base_dir) / "data" / f"{task}_task_info.csv")
    short_rows: list[dict] = []
    condition_rows: list[dict] | None = [] if task == "wcst" else None
    for row in task_info.itertuples(index=False):
        path = os.path.join(base_dir, "data", task, row.filename)
        raw = mne.io.read_raw_snirf(path, verbose="error")
        rename_raw_to_canonical(raw, path)
        short_rows.extend(
            hbo_short_channel_rows(raw, row.subj_id, row.task_id, row.task_order)
        )
        if task == "wcst":
            condition_rows.append(
                wcst_condition_count_row(
                    base_dir, row.subj_id, row.task_id, row.task_order,
                    raw, analyzer_cls,
                )
            )
        logger.info("%s: processed %s (order %s)", task, row.subj_id, row.task_order)
    condition_df = pd.DataFrame(condition_rows) if condition_rows is not None else None
    return pd.DataFrame(short_rows), condition_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Requires BASE_DIR (raw data root) and OUTPUT_DIR (run tree) in the "
               "environment or .env. Refuses protected run trees.",
    )
    parser.parse_args()
    base_dir = os.environ.get("BASE_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")
    if not base_dir or not output_dir:
        logger.error("BASE_DIR and OUTPUT_DIR must be set")
        sys.exit(1)

    refuse_unsafe_output_dir(output_dir)

    out_dir = Path(output_dir) / "04b_session_info"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer_cls = _wcst_analyzer_class()

    for task in ("stroop", "wcst"):
        short_df, condition_df = build_task_session_info(base_dir, task, analyzer_cls)
        short_df.to_csv(out_dir / f"short_channels_{task}.csv", index=False)
        if condition_df is not None:
            condition_df.to_csv(out_dir / "wcst_condition_counts.csv", index=False)

    logger.info("Wrote session info to %s", out_dir)

    write_exclusion_log(output_dir, base_dir)
    logger.info("Wrote low-accuracy exclusion log to %s", out_dir / "low_accuracy_sessions.csv")


if __name__ == "__main__":
    main()
