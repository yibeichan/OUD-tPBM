#!/usr/bin/env python3
"""
Step 3a: Channel-Level Group Analysis.

This script performs group-level statistical analysis at the channel level,
testing whether each channel shows significant activation across the included
baseline sessions (n=12 before any session exclusion). A channel is tested only
when every included session has a value for it.

Statistical Methods (appropriate for n=12):
- One-sample permutation test (10,000 permutations, sign-flipping method)
- Cohen's d effect size with BCa bootstrap 95% CI (5,000 samples)
- FDR correction (Benjamini-Hochberg) for multiple comparisons within contrast

Usage:
    python step3a_channel_group_analysis.py --task stroop
    python step3a_channel_group_analysis.py --task wcst
    python step3a_channel_group_analysis.py --task stroop --contrast stroop_effect

Output:
    OUTPUT_DIR/05_channel_group_analysis/{task}/{task}_{contrast}_channel_statistics.csv
    OUTPUT_DIR/05_channel_group_analysis/{task}/{task}_{contrast}_group_analysis.png
    OUTPUT_DIR/05_channel_group_analysis/{task}/{task}_all_contrasts_channel_statistics.csv
    OUTPUT_DIR/05_channel_group_analysis/{task}/{task}_channel_group_summary.json
"""
import numpy as np
import pandas as pd
import json
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import utilities
# Explicit, rather than relying on the path insert roi_utils happens to do,
# so this import does not depend on module ordering.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
from seeding import DEFAULT_SEED, rng_for_key  # noqa: E402

from roi_utils import (
    get_glm_with_roi_mappings,
    load_roi_mapping,
    read_glm_contrasts,
)

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

# Statistical constants
N_PERMUTATIONS = 10000  # For permutation-based p-values
N_BOOTSTRAP = 5000      # For BCa confidence intervals
ALPHA = 0.05            # Significance level


def bca_bootstrap_ci(values: np.ndarray, statistic_func=np.mean,
                     n_bootstrap: int = N_BOOTSTRAP, alpha: float = ALPHA,
                     *, rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    BCa (bias-corrected and accelerated) bootstrap confidence interval.

    One-sample version: resamples the subject values of a single channel.

    Parameters:
    -----------
    values : array-like
        Data values (for one-sample statistics)
    statistic_func : callable
        Function to compute statistic (default: np.mean)
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (default: 0.05 for 95% CI)
    rng : np.random.Generator, optional
        Generator for the bootstrap draws (default: a generator on DEFAULT_SEED)

    Returns:
    --------
    tuple: (ci_lower, ci_upper)
    """
    rng = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    n = len(values)

    # Original statistic
    stat_obs = statistic_func(values)

    # Bootstrap statistics
    stat_boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        stat_boot = statistic_func(values[idx])
        if not np.isnan(stat_boot):
            stat_boots.append(stat_boot)

    stat_boots = np.array(stat_boots)

    # If not enough valid bootstraps, fall back to percentile
    if len(stat_boots) < 100:
        ci_lower = np.percentile(stat_boots, 100 * alpha/2) if len(stat_boots) > 0 else np.nan
        ci_upper = np.percentile(stat_boots, 100 * (1 - alpha/2)) if len(stat_boots) > 0 else np.nan
        return ci_lower, ci_upper

    # Bias correction
    n_below = np.sum(stat_boots < stat_obs)
    z0 = stats.norm.ppf(n_below / len(stat_boots)) if 0 < n_below < len(stat_boots) else 0

    # Acceleration using jackknife
    stat_jacks = []
    for i in range(n):
        idx = np.concatenate([np.arange(i), np.arange(i+1, n)])
        if len(idx) >= 3:
            stat_jack = statistic_func(values[idx])
            if not np.isnan(stat_jack):
                stat_jacks.append(stat_jack)

    if len(stat_jacks) > 0:
        stat_jacks = np.array(stat_jacks)
        stat_jack_mean = np.mean(stat_jacks)
        numerator = np.sum((stat_jack_mean - stat_jacks)**3)
        denominator = 6 * (np.sum((stat_jack_mean - stat_jacks)**2))**1.5
        a = numerator / denominator if denominator != 0 else 0
    else:
        a = 0

    # BCa adjusted percentiles
    z_alpha2 = stats.norm.ppf(alpha/2)
    z_1alpha2 = stats.norm.ppf(1 - alpha/2)

    denom1 = 1 - a * (z0 + z_alpha2)
    denom2 = 1 - a * (z0 + z_1alpha2)

    if denom1 != 0 and denom2 != 0:
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha2) / denom1)
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1alpha2) / denom2)
    else:
        alpha1 = alpha/2
        alpha2 = 1 - alpha/2

    alpha1 = np.clip(alpha1, 0.001, 0.999)
    alpha2 = np.clip(alpha2, 0.001, 0.999)

    ci_lower = np.percentile(stat_boots, 100 * alpha1)
    ci_upper = np.percentile(stat_boots, 100 * alpha2)

    return ci_lower, ci_upper


def one_sample_permutation_test(values: np.ndarray, n_permutations: int = N_PERMUTATIONS,
                                *, rng: np.random.Generator = None) -> Dict:
    """
    One-sample permutation test testing if mean differs from zero.

    Uses sign-flipping permutation (appropriate for one-sample test).

    Parameters:
    -----------
    values : array-like
        Data values to test (H0: mean = 0)
    n_permutations : int
        Number of permutations
    rng : np.random.Generator, optional
        Generator for the sign flips (default: a generator on DEFAULT_SEED)

    Returns:
    --------
    dict with keys: t_statistic, p_value, mean, se, n
    """
    rng = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    n = len(values)

    if n < 3:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean': np.nan,
            'se': np.nan,
            'n': n
        }

    # Observed t-statistic
    mean_obs = np.mean(values)
    se_obs = stats.sem(values)
    t_obs = mean_obs / se_obs if se_obs > 0 else 0

    # Permutation test: randomly flip signs
    t_perms = []
    for _ in range(n_permutations):
        # Random sign flips
        signs = rng.choice([-1, 1], size=n)
        values_perm = values * signs
        mean_perm = np.mean(values_perm)
        se_perm = stats.sem(values_perm)
        t_perm = mean_perm / se_perm if se_perm > 0 else 0
        t_perms.append(t_perm)

    t_perms = np.array(t_perms)

    # Two-tailed p-value (add-one/Laplace correction: a count of 0 must not
    # floor to an exact 0.0, which would claim infinite evidence)
    n_exceed = np.sum(np.abs(t_perms) >= np.abs(t_obs))
    p_value = (1 + n_exceed) / (1 + n_permutations)

    return {
        't_statistic': float(t_obs),
        'p_value': float(p_value),
        'mean': float(mean_obs),
        'se': float(se_obs),
        'n': n
    }


def compute_cohens_d_with_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                             *, rng: np.random.Generator = None) -> Dict:
    """
    Compute Cohen's d for one-sample test with BCa bootstrap CI.

    For one-sample: d = mean / std

    Parameters:
    -----------
    values : array-like
        Data values
    n_bootstrap : int
        Number of bootstrap samples
    rng : np.random.Generator, optional
        Generator for the bootstrap draws (default: a generator on DEFAULT_SEED)

    Returns:
    --------
    dict with keys: cohens_d, ci_lower, ci_upper
    """
    n = len(values)

    if n < 3:
        return {
            'cohens_d': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }

    # Observed Cohen's d
    mean_obs = np.mean(values)
    std_obs = np.std(values, ddof=1)
    d_obs = mean_obs / std_obs if std_obs > 0 else 0

    # BCa bootstrap CI on Cohen's d
    def cohens_d_func(x):
        m = np.mean(x)
        s = np.std(x, ddof=1)
        return m / s if s > 0 else 0

    ci_lower, ci_upper = bca_bootstrap_ci(values, statistic_func=cohens_d_func,
                                          n_bootstrap=n_bootstrap, rng=rng)

    return {
        'cohens_d': float(d_obs),
        'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else np.nan,
        'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else np.nan
    }


def load_baseline_glm_data(task: str, contrast: str,
                           paired_files: List[Tuple] = None) -> pd.DataFrame:
    """
    Load GLM data for all included baseline sessions (task_order=1).

    Parameters:
    -----------
    task : str
        Task name ('stroop' or 'wcst')
    contrast : str
        Contrast name to extract
    paired_files : list, optional
        Output of ``get_glm_with_roi_mappings(task, baseline_only=True)``;
        looked up when not given.

    Returns:
    --------
    DataFrame with columns: subject, session_id, channel, effect, t_stat, network, network_roi
    """
    # Get baseline GLM files with ROI mappings
    if paired_files is None:
        paired_files = get_glm_with_roi_mappings(task, baseline_only=True)

    if not paired_files:
        logger.error(f"No baseline GLM files found for task={task}")
        return pd.DataFrame()

    logger.info(f"Loading data from {len(paired_files)} baseline sessions...")

    all_data = []

    for glm_file, roi_mapping_file, session_id, subject_id in paired_files:
        # Load GLM results (channel set and contrast masking applied here)
        glm_df = read_glm_contrasts(glm_file, task, session_id)

        # Filter for contrast and HbO
        contrast_data = glm_df[(glm_df['contrast'] == contrast) &
                              (glm_df['chromophore'] == 'hbo')].copy()

        # Every tested channel needs a value from every included session, so
        # a session without this contrast (for WCST, one whose maintenance
        # contrasts are masked) would remove the whole contrast from the
        # channel level without an error.
        if len(contrast_data) == 0:
            raise ValueError(
                f"No data for contrast '{contrast}' in {glm_file.name}; every included "
                "session must have it for the channel-level complete-case test"
            )

        # Load ROI mapping
        roi_mapping = load_roi_mapping(roi_mapping_file)

        # Add ROI info
        contrast_data['network'] = contrast_data['channel'].map(roi_mapping.get('channel_to_network', {}))
        contrast_data['network_roi'] = contrast_data['channel'].map(roi_mapping.get('channel_to_network_roi', {}))

        # Add subject/session info
        contrast_data['subject'] = subject_id
        contrast_data['session_id'] = session_id

        # Select relevant columns
        channel_data = contrast_data[['subject', 'session_id', 'channel', 'effect',
                                      'stat', 'network', 'network_roi']].copy()
        channel_data.columns = ['subject', 'session_id', 'channel', 'effect',
                               't_stat', 'network', 'network_roi']

        all_data.append(channel_data)

    if not all_data:
        logger.error(f"No data loaded for contrast '{contrast}'")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} channel observations from {len(all_data)} subjects")

    return combined_df


def create_channel_matrix(glm_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format GLM data to wide format (subjects x channels).

    Parameters:
    -----------
    glm_data : DataFrame
        Long-format GLM data

    Returns:
    --------
    DataFrame with subjects as rows, channels as columns, values as effects
    """
    # Pivot to wide format
    channel_matrix = glm_data.pivot(index='subject', columns='channel', values='effect')

    # Log data completeness
    n_subjects, n_channels = channel_matrix.shape
    logger.info(f"Channel matrix: {n_subjects} subjects × {n_channels} channels")

    # Check completeness
    completeness = (~channel_matrix.isna()).sum()
    logger.info(f"Channel completeness: min={completeness.min()}, max={completeness.max()}, mean={completeness.mean():.1f}")

    return channel_matrix


def run_channel_group_analysis(task: str, contrast: str, channel_matrix: pd.DataFrame,
                               glm_data: pd.DataFrame, min_subjects: int,
                               *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Run group-level analysis on all channels.

    For each channel with a value from every included participant
    (``min_subjects``):
    - One-sample permutation test (H0: mean effect = 0)
    - Cohen's d with BCa bootstrap CI
    - Add ROI mapping info

    Parameters:
    -----------
    task : str
        Task name
    contrast : str
        Contrast name
    channel_matrix : DataFrame
        Subjects x channels matrix
    glm_data : DataFrame
        Original long-format data for ROI info
    min_subjects : int
        Number of valid values a channel needs to be tested: the number of
        included baseline sessions.
    seed : int
        Base seed. Each channel derives its own generators from the channel's
        identity, so a channel's p-value and CI are the same whether it was
        tested alone, first, or last, and whether or not the other contrasts
        ran in the same invocation.

    Returns:
    --------
    DataFrame with channel-level statistics
    """
    results = []

    n_channels_total = len(channel_matrix.columns)
    logger.info(f"Running group analysis on {n_channels_total} channels...")
    logger.info(f"Requiring all {min_subjects} included participants per channel")

    for i, channel in enumerate(channel_matrix.columns):
        if (i + 1) % 200 == 0:
            logger.info(f"  Processing channel {i+1}/{n_channels_total}...")

        channel_values = channel_matrix[channel].values

        # Check for missing data
        valid_idx = ~np.isnan(channel_values)
        n_valid = valid_idx.sum()

        # Complete-case rule: a channel is tested only if every included
        # participant has a value for it (n_valid == min_subjects).
        if n_valid < min_subjects:
            continue

        values = channel_values[valid_idx]

        # Keyed on the channel's identity, not its position in the loop, and
        # separately per statistic so neither depends on the other's draw count.
        channel_key = f"{task}|{contrast}|{channel}"

        # One-sample permutation test
        perm_result = one_sample_permutation_test(
            values, n_permutations=N_PERMUTATIONS,
            rng=rng_for_key(f"{channel_key}|perm", seed)
        )

        # Cohen's d with CI
        d_result = compute_cohens_d_with_ci(
            values, n_bootstrap=N_BOOTSTRAP,
            rng=rng_for_key(f"{channel_key}|boot", seed)
        )

        # Get ROI info for this channel
        channel_roi_data = glm_data[glm_data['channel'] == channel].iloc[0] if len(glm_data[glm_data['channel'] == channel]) > 0 else None
        network = channel_roi_data['network'] if channel_roi_data is not None else 'Unknown'
        network_roi = channel_roi_data['network_roi'] if channel_roi_data is not None else 'Unknown'

        results.append({
            'channel': channel,
            'contrast': contrast,
            'task': task,
            'n_subjects': n_valid,
            'mean_effect': perm_result['mean'],
            'se': perm_result['se'],
            't_statistic': perm_result['t_statistic'],
            'p_value': perm_result['p_value'],
            'cohens_d': d_result['cohens_d'],
            'd_ci_lower': d_result['ci_lower'],
            'd_ci_upper': d_result['ci_upper'],
            'network': network,
            'network_roi': network_roi
        })

    results_df = pd.DataFrame(results)
    logger.info(f"Analyzed {len(results_df)} channels with complete data (all {min_subjects} included participants)")

    return results_df


def apply_fdr_correction(results_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Apply FDR correction within contrast."""
    results_df = results_df.copy()

    if len(results_df) == 0:
        logger.warning("No results to correct")
        results_df['p_fdr'] = []
        results_df['significant_fdr'] = []
        return results_df

    # FDR correction
    rejected, p_adjusted = fdrcorrection(results_df['p_value'], alpha=alpha)
    results_df['p_fdr'] = p_adjusted
    results_df['significant_fdr'] = rejected

    n_sig = rejected.sum()
    logger.info(f"FDR correction: {n_sig}/{len(results_df)} channels significant (α={alpha})")

    return results_df


def create_visualizations(results_df: pd.DataFrame, output_dir: Path, task: str, contrast: str):
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Volcano plot
    ax = axes[0, 0]
    colors = ['red' if sig else 'gray' for sig in results_df.get('significant_fdr', [False]*len(results_df))]
    ax.scatter(results_df['cohens_d'], -np.log10(results_df['p_value']),
              c=colors, alpha=0.6, s=20)
    ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f'Volcano Plot: {task} - {contrast}')
    ax.legend()

    # 2. Effect size distribution
    ax = axes[0, 1]
    ax.hist(results_df['cohens_d'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('Number of channels')
    ax.set_title('Effect Size Distribution')

    # 3. Network-level summary
    ax = axes[1, 0]
    if 'network' in results_df.columns and results_df['network'].notna().any():
        network_sig = results_df.groupby('network')['significant_fdr'].sum()
        network_total = results_df.groupby('network').size()
        network_pct = (network_sig / network_total * 100).sort_values(ascending=False)

        network_pct.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('% Significant Channels')
        ax.set_xlabel('Network')
        ax.set_title('Significant Channels by Network')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. T-statistic distribution
    ax = axes[1, 1]
    ax.hist(results_df['t_statistic'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('T-statistic')
    ax.set_ylabel('Number of channels')
    ax.set_title('T-Statistic Distribution')

    plt.tight_layout()
    plot_file = output_dir / f'{task}_{contrast}_group_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization: {plot_file}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Channel-level group analysis for baseline fNIRS data'
    )
    parser.add_argument('--task', type=str, required=True, choices=['stroop', 'wcst'],
                       help='Task name')
    parser.add_argument('--contrast', type=str, default=None,
                       help='Specific contrast (default: all contrasts for task)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help='RNG seed for bootstrap and sign-flip resampling')
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    # Define contrasts
    if args.task == 'stroop':
        contrasts = ['stroop_effect', 'mixing_cost', 'incongruent_vs_pure'] if args.contrast is None else [args.contrast]
    else:  # wcst
        contrasts = ['correct_vs_incorrect', 'change_vs_regular', 'learning', 'duration_effect_correct'] if args.contrast is None else [args.contrast]

    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3a: CHANNEL-LEVEL GROUP ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Contrasts: {contrasts}")
    logger.info(f"Statistical method: Permutation testing ({N_PERMUTATIONS} permutations)")
    logger.info(f"Effect size: Cohen's d with BCa bootstrap CI ({N_BOOTSTRAP} samples)")
    logger.info(f"RNG seed: {args.seed}")

    # The included baseline sessions; every tested channel needs all of them.
    paired_files = get_glm_with_roi_mappings(args.task, baseline_only=True)
    if not paired_files:
        logger.error(f"No baseline sessions with GLM output and ROI mappings for task={args.task}")
        sys.exit(1)
    min_subjects = len(paired_files)
    logger.info(f"Minimum subjects: {min_subjects} (all {min_subjects} included participants required)")

    # Create output directory
    output_dir = Path(OUTPUT_DIR) / "05_channel_group_analysis" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each contrast
    all_results = []

    for contrast in contrasts:
        logger.info(f"\n{'-'*70}")
        logger.info(f"Processing contrast: {contrast}")
        logger.info(f"{'-'*70}")

        # Load baseline data
        glm_data = load_baseline_glm_data(args.task, contrast, paired_files)

        if len(glm_data) == 0:
            logger.error(f"No data loaded for {contrast}")
            sys.exit(1)

        # Create channel matrix
        channel_matrix = create_channel_matrix(glm_data)

        # Run group analysis
        results = run_channel_group_analysis(args.task, contrast, channel_matrix, glm_data,
                                             min_subjects, seed=args.seed)

        if len(results) == 0:
            logger.error(f"No channel has data from all {min_subjects} included participants for {contrast}")
            sys.exit(1)

        # Apply FDR correction
        results = apply_fdr_correction(results, alpha=ALPHA)

        all_results.append(results)

        # Save contrast-specific results
        contrast_file = output_dir / f'{args.task}_{contrast}_channel_statistics.csv'
        results.to_csv(contrast_file, index=False)
        logger.info(f"Saved: {contrast_file}")

        # Save significant channels only
        sig_results = results[results['significant_fdr']]
        if len(sig_results) > 0:
            sig_file = output_dir / f'{args.task}_{contrast}_significant_channels.csv'
            sig_results.to_csv(sig_file, index=False)
            logger.info(f"Saved {len(sig_results)} significant channels: {sig_file}")

        # Create visualizations
        create_visualizations(results, output_dir, args.task, contrast)

    # Combine all contrasts
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = output_dir / f'{args.task}_all_contrasts_channel_statistics.csv'
        combined_df.to_csv(combined_file, index=False)

        # Create summary JSON
        summary = {
            'task': args.task,
            'n_contrasts': len(contrasts),
            'contrasts': contrasts,
            'total_channels_tested': int(len(combined_df)),
            'total_significant': int(combined_df['significant_fdr'].sum()),
            'statistical_methods': {
                'permutations': N_PERMUTATIONS,
                'bootstrap_samples': N_BOOTSTRAP,
                'alpha': ALPHA,
                'min_subjects': min_subjects,
                'seed': args.seed
            }
        }

        # Per-contrast summary
        summary['per_contrast'] = []
        for contrast in contrasts:
            contrast_data = combined_df[combined_df['contrast'] == contrast]
            if len(contrast_data) > 0:
                summary['per_contrast'].append({
                    'contrast': contrast,
                    'n_channels': int(len(contrast_data)),
                    'n_significant': int(contrast_data['significant_fdr'].sum()),
                    'mean_effect': float(contrast_data['mean_effect'].mean()),
                    'mean_cohens_d': float(contrast_data['cohens_d'].mean())
                })

        summary_file = output_dir / f'{args.task}_channel_group_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"CHANNEL-LEVEL GROUP ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Task: {args.task}")
        print(f"Total channels tested: {len(combined_df)}")
        print(f"Total significant (FDR): {combined_df['significant_fdr'].sum()}")
        print(f"\nPer-contrast breakdown:")
        for item in summary['per_contrast']:
            print(f"  {item['contrast']}: {item['n_significant']}/{item['n_channels']} significant")
        print(f"\nResults saved to: {output_dir}")
    else:
        logger.error("No results to save!")
        sys.exit(1)


if __name__ == "__main__":
    main()
