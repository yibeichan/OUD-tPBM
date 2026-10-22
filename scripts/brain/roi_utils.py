#!/usr/bin/env python3
"""
ROI Utility Functions for fNIRS Brain Analysis.

This module provides reusable functions for:
1. Session ID extraction from filenames
2. Matching GLM files with their per-session ROI mappings
3. Loading session index and ROI mappings
4. Subject ID standardization

These utilities ensure consistent session-specific mapping across the pipeline.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_variant import load_variant  # noqa: E402
from session_exclusions import excluded_sessions  # noqa: E402
from session_info import short_channels, session_lacks_regular_incorrect  # noqa: E402
from subject_ids import (  # noqa: E402
    extract_session_id as _extract_session_id,
    extract_subject_id as _extract_subject_id,
    standardize_subject_id as _standardize_subject_id,
    sort_subjects,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# WCST contrasts that need regular-incorrect trials. A session with none of
# those trials has no estimate for them, so they are treated as missing.
MAINTENANCE_CONTRASTS = ("correct_vs_incorrect", "change_vs_regular", "learning")


def _current_output_dir() -> Optional[str]:
    """OUTPUT_DIR from the environment at call time, else the import-time value.

    Every lookup in this module resolves the run tree through this function
    (and BASE_DIR through ``_current_base_dir``), so GLM files, mappings,
    the variant, short-channel lists and exclusions always come from the
    same tree.
    """
    return os.environ.get("OUTPUT_DIR") or OUTPUT_DIR


def _current_base_dir() -> Optional[str]:
    """BASE_DIR from the environment at call time, else the import-time value."""
    return os.environ.get("BASE_DIR") or BASE_DIR


def extract_session_id(filename: str) -> str:
    """Extract the session ID from a pipeline filename.

    Handles the SNIRF, derived per-session and session-mapping families.
    See :mod:`subject_ids` for the patterns and for why parsing is anchored
    rather than positional.
    """
    return _extract_session_id(filename)


def extract_subject_id(filename: str) -> str:
    """Extract the canonical subject code from a pipeline filename."""
    return _extract_subject_id(filename)


def load_session_index(mapping_dir: Optional[Path] = None) -> Dict:
    """Load the session index that tracks all per-session mappings.
    
    Args:
        mapping_dir: Directory containing session_index.json
                    (default: OUTPUT_DIR/04_channel_mappings)
    
    Returns:
        Session index dictionary with session info
    """
    if mapping_dir is None:
        mapping_dir = Path(_current_output_dir()) / "04_channel_mappings"
    
    index_file = mapping_dir / "session_index.json"
    
    if not index_file.exists():
        logger.error(f"Session index not found: {index_file}")
        logger.error("Run step2a_glm_to_parcellation.py --per-session first")
        return {}
    
    with open(index_file, 'r') as f:
        return json.load(f)


def get_roi_mapping_for_glm(glm_file: Path, task: str = None,
                           mapping_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the session-specific ROI mapping for a GLM file.
    
    Args:
        glm_file: Path to GLM contrasts CSV file
        task: Task type (will be inferred from path if None)
        mapping_dir: Directory containing ROI mappings
        
    Returns:
        Path to the matching ROI mapping JSON file, or None if not found
    """
    if mapping_dir is None:
        mapping_dir = Path(_current_output_dir()) / "04_channel_mappings"
    
    # Extract session ID from GLM file
    session_id = extract_session_id(glm_file.name)
    
    # Infer task from path if not provided
    if task is None:
        if 'stroop' in str(glm_file).lower():
            task = 'stroop'
        elif 'wcst' in str(glm_file).lower():
            task = 'wcst'
        else:
            logger.warning(f"Could not infer task from path: {glm_file}")
    
    # Load session index to find exact match
    session_index = load_session_index(mapping_dir)
    
    if not session_index or 'sessions' not in session_index:
        logger.warning("Session index is empty or invalid")
        return None
    
    # Find matching session
    for session in session_index.get('sessions', []):
        if session.get('session_id') == session_id and session.get('task') == task:
            roi_mapping_file = session.get('roi_mapping_file')
            if roi_mapping_file:
                roi_path = mapping_dir / roi_mapping_file
                if roi_path.exists():
                    return roi_path
                else:
                    logger.warning(f"ROI mapping file not found: {roi_path}")
    
    # Fallback: Try to find by pattern matching
    logger.debug(f"Session not in index, trying pattern match for session_id={session_id}")
    pattern = f"roi_mapping_{task}_*_{session_id}.json"
    matches = list(mapping_dir.glob(pattern))
    
    if matches:
        logger.debug(f"Found match: {matches[0]}")
        return matches[0]
    
    logger.warning(f"No ROI mapping found for session {session_id} task {task}")
    return None


def load_roi_mapping(mapping_path: Path) -> Dict:
    """Load an ROI mapping JSON file.
    
    Args:
        mapping_path: Path to ROI mapping JSON
        
    Returns:
        ROI mapping dictionary
    """
    with open(mapping_path, 'r') as f:
        return json.load(f)


def load_task_info(task: str) -> Dict[str, int]:
    """Load task_order info from task_info CSV file.
    
    Args:
        task: Task name ('stroop' or 'wcst')
        
    Returns:
        Dictionary mapping session_id to task_order
    """
    import pandas as pd
    
    task_info_file = Path(_current_base_dir()) / "data" / f"{task}_task_info.csv"

    # Without it no session can be identified as baseline, and an empty
    # result would let a stage run on no sessions and exit 0.
    if not task_info_file.exists():
        raise FileNotFoundError(f"Task info file not found: {task_info_file}")
    
    df = pd.read_csv(task_info_file)
    
    # Map task_id (session_id) to task_order
    return dict(zip(df['task_id'], df['task_order']))


def get_baseline_sessions(task: str) -> List[str]:
    """Get session IDs for baseline (task_order=1) sessions only.
    
    Args:
        task: Task name ('stroop' or 'wcst')
        
    Returns:
        List of session IDs with task_order=1
    """
    task_orders = load_task_info(task)
    return [session_id for session_id, order in task_orders.items() if order == 1]


def get_all_glm_files(task: str, glm_dir: Optional[Path] = None) -> List[Path]:
    """Get all GLM files for a task.
    
    Args:
        task: Task name ('stroop' or 'wcst')
        glm_dir: Directory containing GLM files (default: OUTPUT_DIR/03_glm/task)
        
    Returns:
        List of GLM file paths
    """
    if glm_dir is None:
        glm_dir = Path(_current_output_dir()) / "03_glm" / task
    
    if not glm_dir.exists():
        logger.warning(f"GLM directory not found: {glm_dir}")
        return []
    
    return sorted(glm_dir.glob("*_glm_contrasts.csv"))


def read_glm_contrasts(glm_file: Path, task: str, session_id: str) -> pd.DataFrame:
    """Read one session's GLM contrasts with the run's channel and contrast rules.

    Returns the CSV plus a ``channel`` column (``S{Source}_D{Detector}``).
    When the run's variant uses long channels only, the session's short
    channels are dropped. For WCST, when the variant masks sessions without
    regular-incorrect trials and this session is one of them, the rows for
    ``MAINTENANCE_CONTRASTS`` are dropped.
    """
    output_dir = _current_output_dir()
    variant = load_variant(output_dir)

    df = pd.read_csv(glm_file)
    df['channel'] = ('S' + df['Source'].astype(int).astype(str) +
                     '_D' + df['Detector'].astype(int).astype(str))

    n_short = 0
    if variant.channels == "long":
        is_short = df['channel'].isin(short_channels(task, session_id, output_dir))
        n_short = int(is_short.sum())
        df = df[~is_short]

    n_masked = 0
    if (task == "wcst" and variant.mask_wcst_without_regular_incorrect
            and session_lacks_regular_incorrect(session_id, output_dir)):
        is_masked = df['contrast'].isin(MAINTENANCE_CONTRASTS)
        n_masked = int(is_masked.sum())
        df = df[~is_masked]

    if n_short or n_masked:
        df = df.reset_index(drop=True)

    logger.info(
        f"{task} {session_id}: dropped {n_short} short-channel rows, "
        f"{n_masked} masked-contrast rows ({len(df)} rows kept)"
    )
    return df


def get_glm_with_roi_mappings(task: str, baseline_only: bool = False) -> List[Tuple[Path, Path, str, str]]:
    """Get GLM files paired with their ROI mappings.
    
    Args:
        task: Task name ('stroop' or 'wcst')
        baseline_only: If True, only return sessions with task_order=1
        
    Sessions the run's variant excludes (``session_exclusions``) are skipped.

    Returns:
        List of tuples: (glm_path, roi_mapping_path, session_id, subject_id)
    """
    results = []
    glm_files = get_all_glm_files(task)
    excluded = excluded_sessions(task, _current_output_dir(), _current_base_dir())
    
    # Get baseline session IDs if filtering
    baseline_sessions = None
    if baseline_only:
        baseline_sessions = set(get_baseline_sessions(task))
        logger.info(f"Filtering to {len(baseline_sessions)} baseline sessions (task_order=1)")
    
    for glm_file in glm_files:
        session_id = extract_session_id(glm_file.name)
        
        # Skip non-baseline sessions if filtering
        if baseline_only and session_id not in baseline_sessions:
            logger.debug(f"Skipping non-baseline session: {session_id}")
            continue

        if session_id in excluded:
            logger.warning(
                f"Skipping excluded session: subject {extract_subject_id(glm_file.name)}, "
                f"session {session_id}"
            )
            continue
        
        roi_mapping = get_roi_mapping_for_glm(glm_file, task)
        
        # A session without a mapping would drop out of every stage built on
        # this list (and step3a's complete-case bar would shrink with it).
        if not roi_mapping:
            raise FileNotFoundError(
                f"No ROI mapping for {glm_file.name}; run step2a/step2b --per-session"
            )
        subject_id = extract_subject_id(glm_file.name)
        results.append((glm_file, roi_mapping, session_id, subject_id))
    
    return results


def require_one_session_per_subject(df: pd.DataFrame, source) -> None:
    """Raise unless ``df`` holds at most one session per subject.

    The combined ROI file is baseline-only when step3b runs with its default;
    step3c and the confirmatory ROI tests average or test per subject, so a
    file holding several sessions of one subject would silently mix
    timepoints.
    """
    if df.empty:
        return
    per_subject = df.groupby('subject')['session_id'].nunique()
    multi = per_subject[per_subject > 1]
    if not multi.empty:
        raise ValueError(
            f"{source} holds more than one session for {sort_subjects(set(multi.index))}; "
            "rerun step3b_roi_extraction.py without --all-sessions"
        )


def standardize_subject_id(subject_str: str) -> str:
    """Return a subject identifier in canonical ``S<N>`` form."""
    return _standardize_subject_id(subject_str)


def get_unique_subjects(task: str) -> List[str]:
    """Get list of unique subjects for a task.
    
    Args:
        task: Task name ('stroop' or 'wcst')
        
    Returns:
        List of unique subject IDs
    """
    glm_files = get_all_glm_files(task)
    subjects = set()
    
    for glm_file in glm_files:
        subject_id = extract_subject_id(glm_file.name)
        subjects.add(subject_id)

    # Numeric order: sorted() on unpadded codes places S10 before S2.
    return sort_subjects(subjects)


def validate_roi_mapping_coverage(task: str) -> Dict:
    """Validate that all GLM files have corresponding ROI mappings.
    
    Args:
        task: Task name
        
    Returns:
        Validation statistics
    """
    results = {
        'task': task,
        'total_glm_files': 0,
        'with_roi_mapping': 0,
        'missing_mapping': [],
        'coverage_rate': 0.0
    }
    
    glm_files = get_all_glm_files(task)
    results['total_glm_files'] = len(glm_files)
    
    for glm_file in glm_files:
        roi_mapping = get_roi_mapping_for_glm(glm_file, task)
        if roi_mapping:
            results['with_roi_mapping'] += 1
        else:
            results['missing_mapping'].append(glm_file.name)
    
    if results['total_glm_files'] > 0:
        results['coverage_rate'] = results['with_roi_mapping'] / results['total_glm_files']
    
    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing ROI Utilities")
    print("=" * 50)
    
    for task in ['stroop', 'wcst']:
        print(f"\nTask: {task.upper()}")
        
        # Validate coverage
        coverage = validate_roi_mapping_coverage(task)
        print(f"  GLM files: {coverage['total_glm_files']}")
        print(f"  With ROI mapping: {coverage['with_roi_mapping']}")
        print(f"  Coverage: {coverage['coverage_rate']:.1%}")
        
        if coverage['missing_mapping']:
            print(f"  Missing mappings: {coverage['missing_mapping'][:5]}")
        
        # Test getting paired files
        paired = get_glm_with_roi_mappings(task)
        print(f"  Paired GLM-ROI files: {len(paired)}")
        if paired:
            first = paired[0]
            print(f"  Example: {first[0].name} -> {first[1].name}")
