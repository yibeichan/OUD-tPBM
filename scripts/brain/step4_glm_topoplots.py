#!/usr/bin/env python3
"""
Step 4: GLM Topoplots - Channel and ROI Level Visualization

PRIMARY PURPOSE: Create publication-quality topographic brain maps from GLM results
Uses pre-computed group statistics from step3a (channel) and step3c (ROI).

This script:
1. Loads channel-level statistics from step3a output
2. Loads ROI-level statistics from step3c output
3. Creates topoplots: uncorrected (all values) and FDR (only surviving
   channels/ROIs painted, the rest set to 0)
4. Saves statistics and visualizations

Usage:
    python step4_glm_topoplots.py --task stroop
    python step4_glm_topoplots.py --task stroop --contrasts stroop_effect mixing_cost

Output:
    OUTPUT_DIR/08_glm_topoplots/{task}/
        - channel_uncorrected/
        - channel_fdr/
        - roi_uncorrected/
        - roi_fdr/
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
from channel_names import build_canonical_montage  # noqa: E402

# Load environment
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

# =============================================================================
# CONSTANTS
# =============================================================================

# Available contrasts
CONTRAST_MAP = {
    "stroop_effect": "stroop_effect",
    "mixing_cost": "mixing_cost",
    "incongruent_vs_pure": "incongruent_vs_pure",
    "correct_vs_incorrect": "correct_vs_incorrect",
    "change_vs_regular": "change_vs_regular",
    "learning": "learning",
    "duration_effect_correct": "duration_effect_correct",
}

STROOP_CONTRASTS = ["stroop_effect", "mixing_cost", "incongruent_vs_pure"]
WCST_CONTRASTS = [
    "correct_vs_incorrect",
    "change_vs_regular",
    "learning",
    "duration_effect_correct",
]

# =============================================================================
# DATA LOADING
# =============================================================================


def load_channel_statistics(task: str, contrast: str) -> pd.DataFrame:
    """Load channel-level group statistics from step3a output.

    Args:
        task: Task name ('stroop' or 'wcst')
        contrast: Contrast name

    Returns:
        DataFrame with channel statistics
    """
    # step3a outputs: stroop_{contrast}_channel_statistics.csv
    input_file = Path(OUTPUT_DIR) / "05_channel_group_analysis" / task / f"{task}_{contrast}_channel_statistics.csv"

    if not input_file.exists():
        logger.warning(f"Channel statistics file not found: {input_file}")
        return pd.DataFrame()

    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} channel statistics from {input_file.name}")

    return df


def load_roi_statistics(task: str) -> pd.DataFrame:
    """Load ROI-level group statistics from step3c output.

    Args:
        task: Task name ('stroop' or 'wcst')

    Returns:
        DataFrame with ROI statistics (all contrasts)
    """
    # step3c outputs: group_roi_statistics_network_roi.csv
    input_file = Path(OUTPUT_DIR) / "07_roi_analysis" / task / "group_roi_statistics_network_roi.csv"

    if not input_file.exists():
        logger.warning(f"ROI statistics file not found: {input_file}")
        return pd.DataFrame()

    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} ROI statistics from {input_file.name}")

    return df


def load_snirf_montage(task: str) -> tuple:
    """Load the canonical union channel montage for a task's recordings.

    Args:
        task: Task name

    Returns:
        Tuple of (MNE Info object, channel index mapping)
    """
    data_dir = Path(BASE_DIR) / "data" / task
    snirf_files = sorted(data_dir.glob("*.snirf"))

    if not snirf_files:
        raise FileNotFoundError(f"No SNIRF files found in {data_dir}")

    info, ch_mapping = build_canonical_montage(snirf_files)

    logger.info(f"Loaded union montage with {len(ch_mapping)} channels from {len(snirf_files)} recordings")
    return info, ch_mapping


def build_roi_channel_map(channel_stats: pd.DataFrame, task: str) -> Dict[str, List[str]]:
    """Build mapping from ROI to list of channels.

    Args:
        channel_stats: Channel statistics with network_roi column
        task: Task name

    Returns:
        Dict mapping ROI name -> list of channel names
    """
    # Get unique ROIs
    rois = sorted(channel_stats['network_roi'].unique())

    roi_channel_map = {}
    for roi in rois:
        channels = channel_stats[channel_stats['network_roi'] == roi]['channel'].unique().tolist()
        roi_channel_map[roi] = channels

    logger.info(f"Built ROI-channel map for {len(roi_channel_map)} ROIs")
    return roi_channel_map


# =============================================================================
# TOPOPLOT CREATION
# =============================================================================

SIG_COL = 'significant_fdr'


def display_values(stats: pd.DataFrame, value_col: str, fdr_corrected: bool) -> np.ndarray:
    """Return the value painted for each row of ``stats``.

    The uncorrected map paints every row's ``value_col``. The FDR map paints
    only the rows whose ``significant_fdr`` is True and sets the rest to 0, so
    a colour on it means the channel (or ROI) survived correction; the FDR
    map is never the uncorrected map under another title.
    A row whose value is missing stays missing on both maps.
    """
    values = stats[value_col].to_numpy(dtype=float)
    if fdr_corrected:
        if SIG_COL not in stats.columns:
            raise KeyError(f"'{SIG_COL}' is required for the FDR map")
        significant = stats[SIG_COL].fillna(False).astype(bool).to_numpy()
        values = np.where(significant | np.isnan(values), values, 0.0)
    return values


def colour_limit(painted_uncorrected: np.ndarray) -> float:
    """Symmetric colour limit: max |value| over the painted uncorrected values, capped at 4.

    Computed from the channels actually painted on the uncorrected map, and
    shared by the FDR map so the two read on one scale.
    """
    finite = painted_uncorrected[np.isfinite(painted_uncorrected)]
    return min(float(np.max(np.abs(finite))), 4.0) if finite.size else 0.0


def _paint(n_channels: int, rows, values) -> np.ndarray:
    """Plot array: each row's value on its channel indices (rows: iterable of index lists)."""
    plot_values = np.full(n_channels, np.nan)
    for indices, value in zip(rows, values):
        for idx in indices:
            plot_values[idx] = value
    return plot_values


def create_channel_topoplot(stats: pd.DataFrame, info: mne.Info,
                             ch_mapping: Dict[str, int],
                             contrast: str, output_dir: Path,
                             fdr_corrected: bool = False) -> None:
    """Create channel-level topoplot from group statistics.

    Args:
        stats: Group statistics from step3a
        info: MNE Info with channel positions
        ch_mapping: Channel name to index mapping
        contrast: Contrast name
        output_dir: Output directory
        fdr_corrected: If True, channels that do not survive FDR are painted
            as 0 (see ``display_values``)
    """
    sig_col = SIG_COL
    targets = [[ch_mapping[ch]] if ch in ch_mapping else [] for ch in stats['channel']]
    n_channels = len(info["ch_names"])
    limit = colour_limit(_paint(n_channels, targets, display_values(stats, 't_statistic', False)))
    plot_values = _paint(n_channels, targets, display_values(stats, 't_statistic', fdr_corrected))

    # CRITICAL: Filter to only channels with data (for proper interpolation)
    # Create a subset info with only overlapping channels
    valid_mask = ~np.isnan(plot_values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 10:
        logger.warning(f"Only {len(valid_indices)} channels have data - skipping plot")
        return

    # Create subset info for plotting
    info_subset = mne.pick_info(info, valid_indices)
    plot_values_subset = plot_values[valid_indices]

    # Symmetric colour scale shared by the uncorrected and FDR maps
    vlim = (-limit, limit)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    evoked = mne.EvokedArray(plot_values_subset[:, np.newaxis], info_subset, tmin=0)

    im, _ = mne.viz.plot_topomap(
        plot_values_subset, evoked.info, axes=ax, cmap="RdBu_r", vlim=vlim,
        sensors=False, contours=8, show=False, extrapolate="local"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("group t-statistic", fontsize=12)

    # Count significant channels
    n_sig = int(stats[sig_col].sum()) if sig_col in stats.columns else 0
    n_total = len(stats)

    # Title
    correction = ("FDR q < .05; non-surviving channels set to 0" if fdr_corrected
                  else "uncorrected, all channels")
    ax.set_title(f"{contrast.replace('_', ' ').title()}\nChannel-Level ({correction})\n"
                 f"n={stats['n_subjects'].iloc[0] if 'n_subjects' in stats.columns else 'N/A'}, "
                 f"{n_sig}/{n_total} survive FDR",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    suffix = "fdr" if fdr_corrected else "uncorrected"
    output_file = output_dir / f"channel_{contrast}_{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_file.name} ({len(valid_indices)} channels plotted)")


def create_roi_topoplot(stats: pd.DataFrame, info: mne.Info,
                        ch_mapping: Dict[str, int], roi_channel_map: Dict[str, List[str]],
                        contrast: str, output_dir: Path,
                        fdr_corrected: bool = False) -> None:
    """Create an ROI-level topoplot: each ROI's channels painted with one value.

    The painted value is step3c's ``mean_t_statistic``, the mean over
    participants of each participant's mean first-level channel t. It is a
    descriptive summary, not the group test statistic; the group
    one-sample test decides ``significant_fdr``. Only the ROI's channels in
    step3a's complete-coverage set are painted.

    Args:
        stats: ROI-level statistics from step3c
        info: MNE Info with channel positions
        ch_mapping: Channel name to index mapping
        roi_channel_map: ROI -> list of channels mapping
        contrast: Contrast name
        output_dir: Output directory
        fdr_corrected: If True, ROIs whose group test does not survive FDR
            are painted as 0 (see ``display_values``)
    """
    # Filter for this contrast
    stats = stats[stats['contrast'] == contrast].copy()

    if stats.empty:
        logger.warning(f"No ROI statistics for contrast {contrast}")
        return

    sig_col = SIG_COL
    # One value per ROI, painted on the ROI's channels
    targets = [[ch_mapping[ch] for ch in roi_channel_map.get(roi, []) if ch in ch_mapping]
               for roi in stats['network_roi']]
    n_channels = len(info["ch_names"])
    limit = colour_limit(_paint(n_channels, targets, display_values(stats, 'mean_t_statistic', False)))
    plot_values = _paint(n_channels, targets, display_values(stats, 'mean_t_statistic', fdr_corrected))

    # CRITICAL: Filter to only channels with data
    valid_mask = ~np.isnan(plot_values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 10:
        logger.warning(f"Only {len(valid_indices)} channels have data - skipping plot")
        return

    # Create subset info for plotting
    info_subset = mne.pick_info(info, valid_indices)
    plot_values_subset = plot_values[valid_indices]

    # Symmetric colour scale shared by the uncorrected and FDR maps
    vlim = (-limit, limit)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    evoked = mne.EvokedArray(plot_values_subset[:, np.newaxis], info_subset, tmin=0)

    im, _ = mne.viz.plot_topomap(
        plot_values_subset, evoked.info, axes=ax, cmap="RdBu_r", vlim=vlim,
        sensors=False, contours=8, show=False, extrapolate="local"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean first-level t (descriptive)", fontsize=12)

    # Count significant ROIs
    n_sig = int(stats[sig_col].sum()) if sig_col in stats.columns else 0
    n_total = len(stats)

    # Title
    correction = ("FDR q < .05 on the group test; non-surviving ROIs set to 0"
                  if fdr_corrected else "uncorrected, all ROIs")
    n_subj = stats['n_subjects'].iloc[0] if 'n_subjects' in stats.columns else 'N/A'
    ax.set_title(f"{contrast.replace('_', ' ').title()}\nROI-Level ({correction})\n"
                 f"n={n_subj}, {n_sig}/{n_total} ROIs survive FDR (group test)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    suffix = "fdr" if fdr_corrected else "uncorrected"
    output_file = output_dir / f"roi_{contrast}_{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_file.name} ({len(valid_indices)} channels plotted)")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: GLM Topoplots - Channel and ROI Level"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["stroop", "wcst"],
        help="Task to analyze"
    )
    parser.add_argument(
        "--contrasts",
        type=str,
        nargs="+",
        default=None,
        help="Contrasts to visualize (default: all for the task)"
    )
    parser.add_argument(
        "--channel-only",
        action="store_true",
        help="Only generate channel-level topoplots"
    )
    parser.add_argument(
        "--roi-only",
        action="store_true",
        help="Only generate ROI-level topoplots"
    )
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    logger.info("=" * 70)
    logger.info("STEP 4: GLM TOPOPLOTS")
    logger.info("=" * 70)
    logger.info(f"Task: {args.task}")

    # Determine contrasts
    if args.contrasts:
        contrasts = args.contrasts
    else:
        contrasts = STROOP_CONTRASTS if args.task == "stroop" else WCST_CONTRASTS

    logger.info(f"Contrasts: {contrasts}")

    # Create output directories
    base_output = Path(OUTPUT_DIR) / "08_glm_topoplots" / args.task
    output_dirs = {
        "channel_uncorrected": base_output / "channel_uncorrected",
        "channel_fdr": base_output / "channel_fdr",
        "roi_uncorrected": base_output / "roi_uncorrected",
        "roi_fdr": base_output / "roi_fdr",
    }

    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load montage
    logger.info("\nLoading channel montage...")
    info, ch_mapping = load_snirf_montage(args.task)

    # Load ROI statistics (all contrasts, for ROI-channel map)
    logger.info("\nLoading ROI statistics from step3c...")
    roi_stats_all = load_roi_statistics(args.task)

    if roi_stats_all.empty and not args.channel_only:
        logger.error("No ROI statistics found. Run step3c first.")
        return

    # Process each contrast
    for contrast in contrasts:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CONTRAST: {contrast}")
        logger.info(f"{'=' * 60}")

        roi_channel_map = None

        # ========================================================================
        # Channel-level analysis (from step3a)
        # ========================================================================
        if not args.roi_only:
            logger.info("\n--- Channel-Level Analysis (step3a) ---")

            channel_stats = load_channel_statistics(args.task, contrast)

            if channel_stats.empty:
                logger.warning(f"No channel statistics for {contrast}, skipping channel plots")
            else:
                # Build ROI-channel map for ROI plots
                roi_channel_map = build_roi_channel_map(channel_stats, args.task)

                # The uncorrected map paints every channel; the FDR map
                # paints only channels that survive FDR (display_values).
                create_channel_topoplot(
                    channel_stats, info, ch_mapping,
                    contrast, output_dirs["channel_uncorrected"],
                    fdr_corrected=False
                )

                create_channel_topoplot(
                    channel_stats, info, ch_mapping,
                    contrast, output_dirs["channel_fdr"],
                    fdr_corrected=True
                )

                # Save statistics copy
                channel_stats.to_csv(output_dirs["channel_uncorrected"] / f"{contrast}_stats.csv", index=False)
                channel_stats.to_csv(output_dirs["channel_fdr"] / f"{contrast}_stats.csv", index=False)

        # ========================================================================
        # ROI-level analysis (from step3c)
        # ========================================================================
        if not args.channel_only:
            logger.info("\n--- ROI-Level Analysis (step3c) ---")

            if roi_stats_all.empty:
                logger.warning("No ROI statistics available")
            else:
                # --roi-only skips the channel block above, so build the
                # ROI-channel map here from this contrast's channel statistics.
                if roi_channel_map is None:
                    ch_stats = load_channel_statistics(args.task, contrast)
                    if not ch_stats.empty:
                        roi_channel_map = build_roi_channel_map(ch_stats, args.task)
                if roi_channel_map is None:
                    logger.warning(f"No channel statistics for {contrast}; "
                                   "cannot build the ROI-channel map, skipping ROI plots")
                    continue

                # Create topoplots
                create_roi_topoplot(
                    roi_stats_all, info, ch_mapping, roi_channel_map,
                    contrast, output_dirs["roi_uncorrected"],
                    fdr_corrected=False
                )

                create_roi_topoplot(
                    roi_stats_all, info, ch_mapping, roi_channel_map,
                    contrast, output_dirs["roi_fdr"],
                    fdr_corrected=True
                )

                # Save statistics for this contrast
                contrast_roi_stats = roi_stats_all[roi_stats_all['contrast'] == contrast]
                contrast_roi_stats.to_csv(output_dirs["roi_uncorrected"] / f"{contrast}_stats.csv", index=False)
                contrast_roi_stats.to_csv(output_dirs["roi_fdr"] / f"{contrast}_stats.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("STEP 4 COMPLETE")
    logger.info(f"Output saved to: {base_output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
