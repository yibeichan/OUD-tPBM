#!/usr/bin/env python3
"""
Step 3c: Group-Level ROI Activation Testing.

PRIMARY PURPOSE: Test whether each ROI shows significant task-related activation
at the group level (one-sample t-test: activation ≠ 0).

This script:
1. Loads ROI activation from step3b (per-subject files)
2. Computes group-level statistics per ROI
3. Creates visualizations and summary reports

Usage:
    python step3c_roi_group_analysis.py --task stroop
    python step3c_roi_group_analysis.py --task wcst

Output:
    OUTPUT_DIR/07_roi_analysis/{task}/group_roi_statistics_network_roi.csv
    OUTPUT_DIR/07_roi_analysis/{task}/group_roi_run_metadata_network_roi.json
    OUTPUT_DIR/07_roi_analysis/{task}/group_roi_summary.png
"""
import numpy as np
import pandas as pd
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Import utilities
# Explicit, rather than relying on the path insert roi_utils happens to do,
# so this import does not depend on module ordering.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
from seeding import DEFAULT_SEED, rng_for_key  # noqa: E402

from roi_utils import require_one_session_per_subject, standardize_subject_id

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

# Bootstrap settings for small samples
N_BOOTSTRAP = 5000  # Number of bootstrap iterations for 95% CI


def load_roi_activation_data(task: str, roi_level: str = 'network_roi') -> pd.DataFrame:
    """Load combined ROI activation data from step3b.

    step3b already restricts the file to baseline sessions (its default), and
    require_one_session_per_subject fails if a subject has more than one.

    Args:
        task: Task name
        roi_level: ROI aggregation level

    Returns:
        DataFrame with all subjects' ROI activation (one session per subject)
    """
    input_file = Path(OUTPUT_DIR) / "06_roi_activation" / task / f"all_subjects_roi_activation_{roi_level}.csv"

    if not input_file.exists():
        logger.error(f"ROI activation file not found: {input_file}")
        logger.error("Run step3b_roi_extraction.py first")
        return pd.DataFrame()

    df = pd.read_csv(input_file)

    # Standardize subject IDs
    df['subject'] = df['subject'].apply(standardize_subject_id)
    require_one_session_per_subject(df, input_file)

    logger.info(f"Loaded {len(df)} ROI entries from {df['subject'].nunique()} subjects")
    return df


def compute_group_statistics(df: pd.DataFrame, roi_level: str = 'network_roi',
                             *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Compute group-level statistics per ROI.

    Statistical approach for n=12:
    - One-sample t-test (field standard for neuroimaging group analysis)
    - Bootstrap 95% CI for mean activation (quantifies uncertainty)
    - Leave-one-out influence check (identifies if single subjects dominate results)

    Args:
        df: ROI activation DataFrame
        roi_level: ROI column name
        seed: Base seed. Each ROI x contrast cell derives its own generator
            from its identity, so a cell's CI is the same whether it was
            computed alone or alongside every other cell.

    Returns:
        Group statistics DataFrame
    """
    results = []

    for roi in df[roi_level].unique():
        for contrast in df['contrast'].unique():
            roi_data = df[(df[roi_level] == roi) & (df['contrast'] == contrast)]

            # Sorted before the values are taken: the bootstrap draws indices
            # into this array, so its order is part of the result. Row order
            # here is whatever step3b's concat produced, which would otherwise
            # make the published CI depend on file-discovery order rather than
            # on the data.
            roi_data = roi_data.sort_values('subject')

            if len(roi_data) < 3:
                continue

            activations = roi_data['activation'].values
            t_stats = roi_data['t_statistic'].values
            n = len(activations)

            # Group statistics
            result = {
                roi_level: roi,
                'network': roi_data['network'].iloc[0],
                'contrast': contrast,
                'n_subjects': n,
                'mean_activation': np.mean(activations),
                'std_activation': np.std(activations, ddof=1),
                'sem_activation': np.std(activations, ddof=1) / np.sqrt(n),
                'median_activation': np.median(activations),
                'mean_t_statistic': np.mean(t_stats),
            }

            # One-sample t-test against zero
            # Field standard for neuroimaging group analysis; robust for n >= 12
            t_val, p_val = stats.ttest_1samp(activations, 0)
            result['group_t_stat'] = t_val
            result['group_p_value'] = p_val

            # Effect size (Cohen's d)
            # Standard measure in neuroimaging; interpretable across studies
            if np.std(activations) > 0:
                result['cohens_d'] = np.mean(activations) / np.std(activations, ddof=1)
            else:
                result['cohens_d'] = 0

            # Bootstrap 95% CI for the mean activation
            # Quantifies uncertainty without parametric assumptions
            # Essential for small samples where SEM may underestimate true variability
            # Keyed on the cell's identity, not its position in the loop.
            rng = rng_for_key(f"{contrast}|{roi}", seed)
            boot_means = []
            for _ in range(N_BOOTSTRAP):
                boot_sample = rng.choice(activations, size=n, replace=True)
                boot_means.append(np.mean(boot_sample))
            result['mean_ci_lower'] = np.percentile(boot_means, 2.5)
            result['mean_ci_upper'] = np.percentile(boot_means, 97.5)

            # Leave-one-out influence check
            # Quantifies how much any single subject affects the group mean
            # Large influence (>50% of SEM) indicates results are driven by outliers
            loo_means = []
            for i in range(n):
                loo_activations = np.delete(activations, i)
                loo_means.append(np.mean(loo_activations))
            loo_range = np.max(loo_means) - np.min(loo_means)
            result['loo_influence_range'] = loo_range
            # Normalize by SEM for interpretability
            result['loo_influence_sem_ratio'] = loo_range / result['sem_activation'] if result['sem_activation'] > 0 else np.nan

            # Mean channels per subject for this ROI
            result['mean_n_channels'] = roi_data['n_channels'].mean()

            results.append(result)

    results_df = pd.DataFrame(results)
    
    # Apply FDR correction within each contrast
    for contrast in results_df['contrast'].unique():
        mask = results_df['contrast'] == contrast
        if 'group_p_value' in results_df.columns:
            p_vals = results_df.loc[mask, 'group_p_value'].values
            if len(p_vals) > 0 and not np.all(np.isnan(p_vals)):
                valid_mask = ~np.isnan(p_vals)
                rejected, p_fdr = fdrcorrection(p_vals[valid_mask], alpha=0.05)
                p_fdr_full = np.full_like(p_vals, np.nan)
                p_fdr_full[valid_mask] = p_fdr
                results_df.loc[mask, 'p_fdr'] = p_fdr_full
                sig_full = np.full(len(p_vals), False)
                sig_full[valid_mask] = rejected
                results_df.loc[mask, 'significant_fdr'] = sig_full
    
    return results_df


def create_group_visualization(group_stats: pd.DataFrame, output_dir: Path,
                              contrast: str = None):
    """Create visualizations for group-level ROI results.
    
    Args:
        group_stats: Group statistics DataFrame
        output_dir: Output directory
        contrast: Specific contrast to plot
    """
    if contrast:
        data = group_stats[group_stats['contrast'] == contrast]
        suffix = f"_{contrast}"
    else:
        data = group_stats
        suffix = ""
    
    if len(data) == 0:
        logger.warning("No data to visualize")
        return
    
    # 1. Activation by network
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    # Use bootstrap CI if available, otherwise SEM
    if 'mean_ci_lower' in data.columns and 'mean_ci_upper' in data.columns:
        network_data = data.groupby('network').agg({
            'mean_activation': 'mean',
            'mean_ci_lower': 'mean',
            'mean_ci_upper': 'mean'
        }).reset_index()
        network_data['ci_half_width'] = network_data['mean_activation'] - network_data['mean_ci_lower']
        ax.bar(network_data['network'], network_data['mean_activation'],
               yerr=network_data['ci_half_width'],
               capsize=5, alpha=0.7, edgecolor='black', label='Bootstrap 95% CI')
    else:
        network_data = data.groupby('network')['mean_activation'].agg(['mean', 'sem']).reset_index()
        ax.bar(network_data['network'], network_data['mean'], yerr=network_data['sem'],
               capsize=5, alpha=0.7, edgecolor='black', label='SEM')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Network')
    ax.set_ylabel('Mean Activation')
    ax.set_title(f'ROI Activation by Network{" - " + contrast if contrast else ""}')
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Significance heatmap
    ax = axes[1]
    if 'significant_fdr' in data.columns:
        try:
            sig_counts = data.groupby(['network', 'contrast'])['significant_fdr'].sum().unstack(fill_value=0)
            # Ensure numeric type
            sig_counts = sig_counts.astype(int)
            if len(sig_counts) > 0 and sig_counts.sum().sum() > 0:
                sns.heatmap(sig_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
                ax.set_title('Significant ROIs per Network/Contrast')
            else:
                ax.text(0.5, 0.5, 'No significant ROIs\n(FDR < 0.05)', 
                       ha='center', va='center', fontsize=14)
                ax.set_title('Significance Summary')
        except Exception as e:
            ax.text(0.5, 0.5, f'Could not create heatmap:\n{str(e)[:50]}', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('Significance Summary')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"group_roi_summary{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to group_roi_summary{suffix}.png")


def main():
    """Run group-level ROI analysis."""
    parser = argparse.ArgumentParser(
        description='Group-level ROI activation analysis'
    )
    parser.add_argument('--task', type=str, required=True, choices=['stroop', 'wcst'],
                       help='Task name')
    parser.add_argument('--roi-level', type=str, default='network_roi',
                       choices=['network', 'network_roi', 'full_roi'],
                       help='ROI aggregation level (only network_roi feeds step4 and the hypothesis tests)')
    parser.add_argument('--contrast', type=str, default=None,
                       help='Restrict the summary figure to this contrast; statistics are computed for every contrast')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help='RNG seed for bootstrap resampling')
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3c: GROUP-LEVEL ROI ANALYSIS ({args.task.upper()})")
    logger.info(f"{'='*70}")
    logger.info(f"RNG seed: {args.seed}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR) / "07_roi_analysis" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ROI activation data
    roi_df = load_roi_activation_data(args.task, args.roi_level)
    
    if len(roi_df) == 0:
        logger.error("No ROI data loaded!")
        sys.exit(1)
    
    logger.info(f"Loaded data for {roi_df['subject'].nunique()} subjects, "
               f"{roi_df[args.roi_level].nunique()} ROIs, "
               f"{roi_df['contrast'].nunique()} contrasts")
    
    # Compute group statistics
    logger.info("\nComputing group statistics...")
    group_stats = compute_group_statistics(roi_df, args.roi_level, seed=args.seed)
    
    # Save group statistics
    group_stats_file = output_dir / f"group_roi_statistics_{args.roi_level}.csv"
    group_stats.to_csv(group_stats_file, index=False)
    logger.info(f"Saved group statistics to {group_stats_file}")

    # The seed governs every bootstrap CI in that CSV, so it travels with it
    # rather than only reaching the run log.
    run_metadata_file = output_dir / f"group_roi_run_metadata_{args.roi_level}.json"
    with open(run_metadata_file, 'w') as f:
        json.dump({
            'task': args.task,
            'roi_level': args.roi_level,
            'seed': args.seed,
            'n_bootstrap': N_BOOTSTRAP,
        }, f, indent=2)
    logger.info(f"Saved run metadata to {run_metadata_file}")
    
    # Report significant ROIs
    if 'significant_fdr' in group_stats.columns:
        sig_rois = group_stats[group_stats['significant_fdr'] == True]
        logger.info(f"\nSignificant ROIs (FDR < 0.05): {len(sig_rois)}")
        if len(sig_rois) > 0:
            for _, row in sig_rois.iterrows():
                logger.info(f"  {row[args.roi_level]} ({row['contrast']}): "
                          f"d={row['cohens_d']:.2f}, p_fdr={row['p_fdr']:.4f}")

    # Warn about high-influence ROIs (results may be driven by single subjects)
    if 'loo_influence_sem_ratio' in group_stats.columns:
        high_influence = group_stats[group_stats['loo_influence_sem_ratio'] > 0.5]
        if len(high_influence) > 0:
            logger.warning(f"\n{len(high_influence)} ROIs with high subject influence (ratio > 0.5):")
            for _, row in high_influence.iterrows():
                logger.warning(f"  {row[args.roi_level]} ({row['contrast']}): "
                             f"influence/SEM = {row['loo_influence_sem_ratio']:.2f}")
            logger.warning("  Consider individual subject inspection for these ROIs.")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_group_visualization(group_stats, output_dir, args.contrast)
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Subjects analyzed:   {roi_df['subject'].nunique()}")
    logger.info(f"Total ROIs:          {group_stats[args.roi_level].nunique()}")
    logger.info(f"Networks:            {group_stats['network'].nunique()}")
    logger.info(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
