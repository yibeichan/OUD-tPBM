#!/usr/bin/env python3
"""
Step 3b: ROI Extraction per Subject.

This script:
1. Loads GLM results for each subject
2. Uses SESSION-SPECIFIC ROI mappings to assign channels to ROIs
3. Aggregates channel-level activation to ROI-level
4. Outputs per-subject ROI activation files

This ensures correct channel-to-ROI mapping for each session, accounting for
channel counts that vary across sessions.

Usage:
    python step3b_roi_extraction.py --task stroop
    python step3b_roi_extraction.py --task wcst
    python step3b_roi_extraction.py --task stroop --subject S1   # debugging

Output:
    OUTPUT_DIR/06_roi_activation/{task}/{subject}_{session}_roi_activation.csv
    OUTPUT_DIR/06_roi_activation/{task}/all_subjects_roi_activation_network_roi.csv
    OUTPUT_DIR/06_roi_activation/{task}/extraction_stats_network_roi.csv

With --subject, the combined and stats files get a ``_subject_S<N>`` suffix, so
a debugging run never replaces the files step3c and the hypothesis tests read.

Columns ``t_statistic`` and ``p_value`` in the ROI files are the mean (or
median) of the channels' first-level t and p values. They are descriptive
only: a mean of p-values has no inferential meaning, and no later stage reads
them. The ROI test statistic is computed from ``activation`` in step3c.
"""
import numpy as np
import pandas as pd
import json
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import utilities
from roi_utils import (
    get_glm_with_roi_mappings,
    load_roi_mapping,
    read_glm_contrasts,
    extract_subject_id,
    extract_session_id,
    standardize_subject_id
)
from analysis_variant import VariantError, refuse_protected_tree

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if not BASE_DIR or not OUTPUT_DIR:
    print("Error: BASE_DIR and OUTPUT_DIR must be set in .env file")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_glm_results(glm_file: Path, task: str, session_id: str) -> pd.DataFrame:
    """Load one session's GLM results with the run's channel and contrast rules.

    Args:
        glm_file: Path to GLM contrasts CSV
        task: Task name
        session_id: Session the file belongs to

    Returns:
        DataFrame with GLM results and a ``channel`` column
    """
    return read_glm_contrasts(glm_file, task, session_id)


def add_roi_info_to_glm(df: pd.DataFrame, roi_mapping: Dict) -> pd.DataFrame:
    """Add ROI information to GLM results using session-specific mapping.
    
    Args:
        df: GLM results DataFrame
        roi_mapping: Session-specific ROI mapping dictionary
        
    Returns:
        DataFrame with ROI columns added
    """
    df = df.copy()
    
    # Add ROI mappings
    channel_to_network = roi_mapping.get('channel_to_network', {})
    channel_to_network_roi = roi_mapping.get('channel_to_network_roi', {})
    channel_to_full_roi = roi_mapping.get('channel_to_full_roi', {})
    channel_to_parcel_idx = roi_mapping.get('channel_to_parcel_idx', {})
    
    df['network'] = df['channel'].map(channel_to_network)
    df['network_roi'] = df['channel'].map(channel_to_network_roi)
    df['full_roi'] = df['channel'].map(channel_to_full_roi)
    df['parcel_idx'] = df['channel'].map(channel_to_parcel_idx)
    
    # Mark channels with ROI assignments
    df['has_roi'] = df['network'].notna()

    return df


def channel_mapping_counts(glm_with_roi: pd.DataFrame) -> Tuple[int, int]:
    """Return (HbO channels, HbO channels with an ROI), counting each channel once.

    The GLM frame has one row per channel per contrast, so counting rows
    multiplied both numbers by the number of contrasts.
    """
    hbo = glm_with_roi[glm_with_roi['chromophore'] == 'hbo']
    n_total = int(hbo['channel'].nunique())
    n_mapped = int(hbo.loc[hbo['has_roi'], 'channel'].nunique())
    return n_total, n_mapped


def aggregate_to_roi(df: pd.DataFrame, contrast_names: List[str],
                     roi_level: str = 'network_roi',
                     method: str = 'mean') -> pd.DataFrame:
    """Aggregate channel-level activation to ROI level.

    Statistical note for n=12: Simple mean is recommended because:
    1. Weighted schemes using effect-size-dependent weights (e.g., |t-stat|) introduce
       circular reasoning - channels with larger effects get more weight, biasing results
    2. For small samples, unbiased aggregation is essential for valid inference
    3. The simple mean is statistically neutral and preserves variance information

    Args:
        df: GLM results with ROI info
        contrast_names: Contrasts to process
        roi_level: Aggregation level ('network', 'network_roi', 'full_roi')
        method: Aggregation method ('mean', 'median')

    Returns:
        ROI-aggregated results
    """
    # Filter to HbO channels with ROI assignments
    df_roi = df[(df['has_roi']) & (df['chromophore'] == 'hbo')].copy()

    if len(df_roi) == 0:
        logger.warning("No channels mapped to ROIs!")
        return pd.DataFrame()

    roi_results = []

    for roi in df_roi[roi_level].unique():
        if pd.isna(roi):
            continue

        roi_data = df_roi[df_roi[roi_level] == roi]

        # Get network for this ROI
        network = roi_data['network'].iloc[0] if len(roi_data) > 0 else None

        for contrast in contrast_names:
            contrast_data = roi_data[roi_data['contrast'] == contrast]

            if len(contrast_data) == 0:
                continue

            result = {
                roi_level: roi,
                'network': network,
                'contrast': contrast,
                'n_channels': len(contrast_data),
            }

            # Get values
            effects = contrast_data['effect'].values
            t_stats = contrast_data['stat'].values
            p_values = contrast_data['p_value'].values

            # Aggregate based on method
            if method == 'mean':
                result['activation'] = np.mean(effects)
                result['activation_std'] = np.std(effects, ddof=1) if len(effects) > 1 else 0
                # Mean first-level t and p, descriptive only: not ROI test results,
                # not used as weights, and read by no later stage.
                result['t_statistic'] = np.mean(t_stats)
                result['p_value'] = np.mean(p_values)

            elif method == 'median':
                result['activation'] = np.median(effects)
                result['activation_std'] = np.median(np.abs(effects - result['activation']))  # MAD
                result['t_statistic'] = np.median(t_stats)
                result['p_value'] = np.median(p_values)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'mean' or 'median'.")

            # Add channel list for reference
            result['channels'] = ','.join(contrast_data['channel'].tolist())

            roi_results.append(result)

    return pd.DataFrame(roi_results)


def process_single_subject(glm_file: Path, roi_mapping_file: Path,
                          task: str, contrast_names: List[str],
                          roi_level: str = 'network_roi') -> Tuple[pd.DataFrame, Dict]:
    """Process a single subject's GLM file with their session-specific ROI mapping.
    
    Args:
        glm_file: Path to GLM contrasts CSV
        roi_mapping_file: Path to session-specific ROI mapping
        task: Task name
        contrast_names: Contrasts to process
        roi_level: ROI aggregation level
        
    Returns:
        Tuple of (ROI DataFrame, processing stats)
    """
    subject_id = extract_subject_id(glm_file.name)
    session_id = extract_session_id(glm_file.name)
    
    # Load GLM results
    glm_df = load_glm_results(glm_file, task, session_id)
    
    # Load session-specific ROI mapping
    roi_mapping = load_roi_mapping(roi_mapping_file)
    
    # Add ROI info
    glm_with_roi = add_roi_info_to_glm(glm_df, roi_mapping)
    
    # Calculate mapping stats (distinct HbO channels, not rows)
    n_total, n_mapped = channel_mapping_counts(glm_with_roi)
    
    # Aggregate to ROI
    roi_df = aggregate_to_roi(glm_with_roi, contrast_names, roi_level=roi_level)
    
    if len(roi_df) > 0:
        roi_df['subject'] = subject_id
        roi_df['session_id'] = session_id
        roi_df['task'] = task
    
    stats = {
        'subject': subject_id,
        'session_id': session_id,
        'total_channels': n_total,
        'mapped_channels': n_mapped,
        'mapping_rate': n_mapped / n_total if n_total > 0 else 0,
        'n_rois': roi_df[roi_level].nunique() if len(roi_df) > 0 else 0,
        'n_contrasts': len(contrast_names)
    }
    
    return roi_df, stats


def main():
    """Extract ROI-level activation for all subjects."""
    parser = argparse.ArgumentParser(
        description='Extract ROI activation per subject using session-specific mappings'
    )
    parser.add_argument('--task', type=str, required=True, choices=['stroop', 'wcst'],
                       help='Task name')
    parser.add_argument('--subject', type=str, default=None,
                       help='Debugging: process one subject only; outputs get a _subject_S<N> suffix')
    parser.add_argument('--roi-level', type=str, default='network_roi',
                       choices=['network', 'network_roi', 'full_roi'],
                       help='ROI aggregation level. Only the default, network_roi, feeds '
                            'step3c, step4 and the hypothesis tests; other levels write '
                            'files that no later stage reads')
    parser.add_argument('--method', type=str, default='mean',
                       choices=['mean', 'median'],
                       help='Aggregation method (the analysis uses mean; median is for inspection)')
    # Baseline-only is the default: step3c and the confirmatory ROI
    # hypothesis tests read the combined file this writes and assume one
    # baseline session per subject. --all-sessions is for inspection only.
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument('--baseline-only', dest='baseline_only', action='store_true',
                          default=True,
                          help='Only use baseline sessions (task_order=1); the default')
    sessions.add_argument('--all-sessions', dest='baseline_only', action='store_false',
                          help='Use every session; the combined file is then not a '
                               'valid input to step3c or the ROI hypothesis tests')
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    
    # Define contrasts based on task (actual contrast names from GLM output)
    if args.task == 'stroop':
        contrast_names = ['stroop_effect', 'mixing_cost', 'incongruent_vs_pure']
    else:  # wcst
        contrast_names = ['correct_vs_incorrect', 'change_vs_regular', 'learning', 'duration_effect_correct']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3b: ROI EXTRACTION ({args.task.upper()})")
    logger.info(f"{'='*70}")
    logger.info(f"Contrasts: {contrast_names}")
    logger.info(f"ROI level: {args.roi_level}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Baseline only: {args.baseline_only}")
    
    # Get GLM files with their matching ROI mappings
    paired_files = get_glm_with_roi_mappings(args.task, baseline_only=args.baseline_only)
    
    if not paired_files:
        logger.error("No GLM files with ROI mappings found!")
        logger.error("Ensure step2a and step2b have been run")
        sys.exit(1)
    
    logger.info(f"\nFound {len(paired_files)} GLM files with ROI mappings")
    
    # Filter for specific subject if requested
    if args.subject:
        subject_std = standardize_subject_id(args.subject)
        paired_files = [(g, r, s, subj) for g, r, s, subj in paired_files 
                       if standardize_subject_id(subj) == subject_std]
        logger.info(f"Filtered to {len(paired_files)} files for subject {subject_std}")
    
    # A single-subject run must not replace the combined files later stages read.
    out_suffix = f"_subject_{subject_std}" if args.subject else ""

    # Create output directory
    output_dir = Path(OUTPUT_DIR) / "06_roi_activation" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each subject
    all_results = []
    all_stats = []
    
    logger.info(f"\nProcessing subjects...")
    
    for glm_file, roi_mapping_file, session_id, subject_id in paired_files:
        logger.info(f"  {subject_id} ({session_id})...")
        
        try:
            roi_df, stats = process_single_subject(
                glm_file, roi_mapping_file, args.task,
                contrast_names, args.roi_level
            )
            
            if len(roi_df) == 0:
                raise ValueError(f"No ROI results for {subject_id} ({session_id})")

            all_results.append(roi_df)
            all_stats.append(stats)

            # Save individual file
            output_file = output_dir / f"{subject_id}_{session_id}_roi_activation.csv"
            roi_df.to_csv(output_file, index=False)

            logger.info(f"    ✓ {stats['mapped_channels']}/{stats['total_channels']} channels, "
                      f"{stats['n_rois']} ROIs → {output_file.name}")

        # A session that fails would vanish from the combined file, and
        # step3c and the confirmatory tests would run one participant short.
        except Exception as e:
            logger.error(f"    ✗ Error: {e}")
            raise

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined file
        combined_file = output_dir / f"all_subjects_roi_activation_{args.roi_level}{out_suffix}.csv"
        combined_df.to_csv(combined_file, index=False)
        
        # Save stats
        stats_df = pd.DataFrame(all_stats)
        stats_file = output_dir / f"extraction_stats_{args.roi_level}{out_suffix}.csv"
        stats_df.to_csv(stats_file, index=False)
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("EXTRACTION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Subjects processed:  {len(all_stats)}")
        logger.info(f"Total ROI entries:   {len(combined_df)}")
        logger.info(f"Unique ROIs:         {combined_df[args.roi_level].nunique()}")
        logger.info(f"Unique networks:     {combined_df['network'].nunique()}")
        logger.info(f"Contrasts:           {combined_df['contrast'].nunique()}")
        logger.info(f"\nAvg mapping rate:    {stats_df['mapping_rate'].mean():.1%}")
        logger.info(f"\nOutput files:")
        logger.info(f"  Combined: {combined_file}")
        logger.info(f"  Stats:    {stats_file}")
        
        # Show network distribution
        logger.info(f"\nROI Distribution by Network:")
        network_counts = combined_df.groupby('network')[args.roi_level].nunique()
        for network, count in sorted(network_counts.items()):
            logger.info(f"  {network}: {count} ROIs")
    else:
        logger.error("No results to save!")


if __name__ == "__main__":
    main()
