#!/usr/bin/env python3
"""Extract ROI activations across ALL timepoints for longitudinal analysis.

The cross-sectional pipeline writes baseline-only ROI activations to
OUTPUT_DIR/06_roi_activation/{task}/ (the cross-sectional
analysis uses baseline sessions only). For longitudinal convergence analysis we also need
post and followup sessions, including the contrasts the unified-EF
classifier requires (stroop_effect, mixing_cost, change_vs_regular,
learning).

This script reuses step3b's per-subject extraction helpers but writes its
combined output to a longitudinal-specific directory so the cross-sectional
files remain canonical.

Output:
    OUTPUT_DIR/10_longitudinal/roi_activations_alltimepoints/{task}/
        all_subjects_roi_activation_network_roi.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "brain"))

from roi_utils import get_glm_with_roi_mappings  # noqa: E402
from step3b_roi_extraction import process_single_subject  # noqa: E402
from analysis_variant import VariantError, refuse_protected_tree  # noqa: E402

load_dotenv()
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


TASK_CONTRASTS = {
    "stroop": ["stroop_effect", "mixing_cost", "incongruent_vs_pure"],
    "wcst": ["correct_vs_incorrect", "change_vs_regular", "learning",
             "duration_effect_correct"],
}


def extract(task: str, roi_level: str = "network_roi") -> None:
    contrasts = TASK_CONTRASTS[task]
    out_dir = OUTPUT_DIR / "10_longitudinal" / "roi_activations_alltimepoints" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = get_glm_with_roi_mappings(task, baseline_only=False)
    log.info("[%s] %d (subject, session) pairs across all timepoints", task, len(paired))

    rows = []
    failed = []
    for glm_file, roi_mapping_file, session_id, subject_id in paired:
        try:
            roi_df, _ = process_single_subject(
                glm_file, roi_mapping_file, task, contrasts, roi_level
            )
        except (VariantError, FileNotFoundError, KeyError) as exc:
            log.error("  %s/%s failed: %s", subject_id, session_id, exc)
            raise
        except Exception as exc:
            # Keep going so every failing session is logged, then fail below.
            log.error("  %s/%s failed: %s", subject_id, session_id, exc)
            failed.append(f"{subject_id}/{session_id}")
            continue
        if roi_df is None or roi_df.empty:
            log.error("  %s/%s: no ROI rows", subject_id, session_id)
            failed.append(f"{subject_id}/{session_id}")
            continue
        rows.append(roi_df)

    # A missing session would silently remove that subject's brain level
    # from the longitudinal classification, so no partial file is written.
    if failed:
        raise RuntimeError(f"[{task}] {len(failed)} session(s) failed: {failed}")
    if not rows:
        raise RuntimeError(f"[{task}] No ROI activations produced")

    combined = pd.concat(rows, ignore_index=True)
    combined_path = out_dir / f"all_subjects_roi_activation_{roi_level}.csv"
    combined.to_csv(combined_path, index=False)
    log.info("[%s] wrote %d rows -> %s", task, len(combined), combined_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["stroop", "wcst", "both"], default="both")
    parser.add_argument("--roi-level", default="network_roi",
                        choices=["network", "network_roi", "full_roi"],
                        help="ROI level. Only network_roi is read by the longitudinal "
                             "stages; other levels write files nothing reads")
    args = parser.parse_args()
    try:
        refuse_protected_tree(str(OUTPUT_DIR))
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    tasks = ["stroop", "wcst"] if args.task == "both" else [args.task]
    for t in tasks:
        extract(t, args.roi_level)


if __name__ == "__main__":
    main()
