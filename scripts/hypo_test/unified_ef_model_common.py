#!/usr/bin/env python3
"""
Unified EF Model - Common Utilities

Shared functions for ROI-level and Channel-level analyses.
This module contains:
- Data loading functions
- Statistical test functions
- Visualization functions
- Helper functions

"""

import copy
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import zscore
from statsmodels.stats.multitest import fdrcorrection
import sys
import os

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'brain'))

from subject_ids import standardize_subject_id

from utils.statistical_methods import (
    setup_plotting, create_scatter_plot,
    load_assessment_data, load_behavioral_data,
    # Seeding is defined in scripts/seeding.py and re-exported here, so every
    # stage reaches one constant and one derivation.
    DEFAULT_SEED, rng_for_key, _resolve_rng
)

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")

logger = logging.getLogger(__name__)


# =============================================================================
# HYPOTHESIS DEFINITIONS (shared by the ROI and channel stages)
# =============================================================================

# H2's ``behavioral_measure`` here is only a default; main() replaces it with
# the run's analysis variant (see resolve_hypotheses below).
PRIMARY_HYPOTHESES = {
    'H1': {
        'name': 'Inhibition',
        'brief_measure': 'Inhibit_t',
        'task': 'stroop',
        'contrast': 'stroop_effect',
        'target_rois': ['SalVentAttn_PFCl', 'Cont_PFCl'],
        'behavioral_measure': 'interference',
    },
    'H2': {
        'name': 'Cognitive Flexibility',
        'brief_measure': 'Shift_t',
        'task': 'wcst',
        'contrast': 'change_vs_regular',
        'target_rois': ['Cont_PFCl', 'Cont_Par'],
        'behavioral_measure': 'perseverative_errors',
    },
    'H3': {
        'name': 'Rule Learning',
        'brief_measure': 'Working Memory_t',
        'task': 'wcst',
        'contrast': 'learning',
        'target_rois': ['Cont_PFCl', 'Cont_Par'],
        'behavioral_measure': 'wm_maintenance_decline',
    },
    'H4': {
        'name': 'Attentional Stability',
        'brief_measure': 'Working Memory_t',
        'task': 'stroop',
        'contrast': 'mixing_cost',
        'target_rois': ['Cont_PFCl'],
        'behavioral_measure': 'rt_variability',
    },
    'H5': {
        'name': 'Global Executive Function',
        'brief_measure': 'GEC_t',
        'task': None,
        'contrast': None,
        'target_rois': [],
        'behavioral_measure': 'global_performance_z',
    },
}


# =============================================================================
# ANALYSIS VARIANT
# =============================================================================

def resolve_hypotheses(hypotheses: dict, variant) -> dict:
    """Return a deep copy of ``hypotheses`` with H2's measure set from ``variant``.

    Only ``hypotheses['H2']['behavioral_measure']`` is replaced, with
    ``variant.h2_metric``; every other hypothesis is returned unchanged. The
    ``PRIMARY_HYPOTHESES`` keeps ``'perseverative_errors'`` as its default,
    so callers that read it directly (tests) see a
    stable value; only ``main()`` resolves it against the run's variant.
    """
    resolved = copy.deepcopy(hypotheses)
    resolved['H2']['behavioral_measure'] = variant.h2_metric
    return resolved


# H5's WCST component: the proportion of rule periods mastered, not the
# count of categories completed, which tracks the number of trials a
# participant got through.
H5_WCST_METRIC = 'prop_rule_periods_mastered'


def global_performance_z(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Composite z-score of Stroop accuracy and the WCST proportion of rule periods mastered.

    ``perf_df`` must carry ``subject_id``, ``overall_accuracy`` and
    ``prop_rule_periods_mastered`` (``H5_WCST_METRIC``). ``scipy.stats.zscore`` returns an all-NaN array
    when its input contains any NaN, so scoring the full column would blank
    the composite for every subject as soon as one subject was missing a
    metric. Dropping NaN rows before scoring keeps the score for everyone
    else; the caller's left merge back onto the full subject list then gives
    the dropped subject(s) NaN rather than a fabricated value.

    Returns a DataFrame with columns ``subject_id``, ``global_performance_z``.
    """
    clean = perf_df.dropna(subset=['overall_accuracy', H5_WCST_METRIC]).copy()
    stroop_z = zscore(clean['overall_accuracy'])
    wcst_z = zscore(clean[H5_WCST_METRIC])
    clean['global_performance_z'] = (stroop_z + wcst_z) / 2
    return clean[['subject_id', 'global_performance_z']]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load and prepare all data (shared by ROI and Channel analyses).

    Returns:
        dict with: assessment, behavior_stroop, behavior_wcst,
                 roi_stroop, roi_wcst, merged
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Assessment
    print("\n[1/6] BRIEF-A (T-scores)...")
    assess_df = load_assessment_data()
    # standardize rather than rewrite the separator: a stray non-canonical
    # spelling must raise here, not slip through and miss the merges below.
    assess_df['subject'] = assess_df['subject_id'].apply(standardize_subject_id)

    # Handle column naming
    if 'Working Memory_t' not in assess_df.columns:
        assess_df.rename(columns={'Working_Memory_t': 'Working Memory_t'}, inplace=True)

    print(f"  Loaded {len(assess_df)} subjects")

    # Behavioral (names are now theoretically meaningful at source)
    print("\n[2/6] Behavioral data...")
    stroop_df, wcst_df = load_behavioral_data()

    print(f"  Stroop: {len(stroop_df)} subjects")
    print(f"  WCST: {len(wcst_df)} subjects")

    # ROI activation (pre-aggregated)
    print("\n[3/6] ROI activation (Stroop)...")
    roi_stroop = _load_roi_activation('stroop')
    print(f"  Loaded {len(roi_stroop)} observations")

    print("\n[4/6] ROI activation (WCST)...")
    roi_wcst = _load_roi_activation('wcst')
    print(f"  Loaded {len(roi_wcst)} observations")

    # Global performance z-score
    print("\n[5/6] Computing global performance z...")
    # Read directly: substituting another accuracy would change what H5 tests.
    stroop_acc = stroop_df[['subject_id', 'overall_accuracy']].copy()

    wcst_cat = wcst_df[['subject_id', H5_WCST_METRIC]].copy()
    merged_perf = stroop_acc.merge(wcst_cat, on='subject_id', how='inner')
    merged_perf = global_performance_z(merged_perf)

    print(f"  Computed for {len(merged_perf)} subjects")

    # Merge all
    print("\n[6/6] Merging all datasets...")
    merged = assess_df[['subject', 'Inhibit_t', 'Shift_t', 'Working Memory_t', 'GEC_t']].copy()

    stroop_behav = stroop_df[['subject_id', 'interference', 'rt_variability']].copy()
    stroop_behav['subject'] = stroop_behav['subject_id'].apply(standardize_subject_id)
    merged = merged.merge(stroop_behav[['subject', 'interference', 'rt_variability']],
                         on='subject', how='left')

    wcst_behav = wcst_df[['subject_id', 'perseverative_errors', 'perseverative_error_rate',
                          'wm_maintenance_decline']].copy()
    wcst_behav['subject'] = wcst_behav['subject_id'].apply(standardize_subject_id)
    merged = merged.merge(wcst_behav[['subject', 'perseverative_errors', 'perseverative_error_rate',
                                     'wm_maintenance_decline']],
                         on='subject', how='left')

    merged_perf['subject'] = merged_perf['subject_id'].apply(standardize_subject_id)
    merged = merged.merge(merged_perf[['subject', 'global_performance_z']],
                         on='subject', how='left')

    print(f"\n  Final dataset: {len(merged)} subjects")

    return {
        'assessment': assess_df,
        'behavior_stroop': stroop_df,
        'behavior_wcst': wcst_df,
        'roi_stroop': roi_stroop,
        'roi_wcst': roi_wcst,
        'merged': merged
    }


def _load_roi_activation(task: str) -> pd.DataFrame:
    """Load pre-aggregated ROI activation data (baseline, one session per subject).

    A missing file raises: without it the task's hypotheses would drop out
    and the confirmatory FDR family would shrink.
    """
    from roi_utils import require_one_session_per_subject

    input_file = Path(OUTPUT_DIR) / "06_roi_activation" / task / "all_subjects_roi_activation_network_roi.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"ROI activation file not found: {input_file}; run step3b first")

    df = pd.read_csv(input_file)
    df['subject'] = df['subject'].apply(standardize_subject_id)
    require_one_session_per_subject(df, input_file)

    return df


def load_channel_data(task: str, contrast: str, cache: dict = None) -> pd.DataFrame:
    """Load RAW channel-level data with caching.

    This is the key difference from ROI:
    - ROI: Pre-aggregated (mean across channels)
    - Channel: Raw per-channel values

    Args:
        task: 'stroop' or 'wcst'
        contrast: Contrast name
        cache: Optional cache dict to store loaded data

    Returns:
        DataFrame with columns: subject, channel, activation, network, network_roi
    """
    # Check cache first
    cache_key = f"{task}_{contrast}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    from roi_utils import (
        standardize_subject_id, get_glm_with_roi_mappings,
        load_roi_mapping, read_glm_contrasts
    )

    glm_roi_pairs = get_glm_with_roi_mappings(task, baseline_only=True)

    # No sessions would leave the task's hypotheses out of the channel-level
    # families without an error.
    if not glm_roi_pairs:
        raise FileNotFoundError(f"No baseline GLM/ROI-mapping pairs for task={task}")

    all_data = []
    for glm_file, roi_mapping_file, session_id, subject_id in glm_roi_pairs:
        subject_std = standardize_subject_id(subject_id)

        # An unreadable CSV raises, as step3b does on the same inputs, rather
        # than dropping the session from every channel test.
        df = read_glm_contrasts(glm_file, task, session_id)

        # Filter for specific contrast and HbO
        contrast_data = df[(df['contrast'] == contrast) & (df['chromophore'] == 'hbo')].copy()

        # A session can lack a contrast legitimately (a WCST session whose
        # maintenance contrasts are masked); say so rather than drop it silently.
        if len(contrast_data) == 0:
            logger.warning(f"{task} {session_id} ({subject_std}): no HbO rows for "
                           f"contrast '{contrast}'; session left out of its channel tests")
            continue

        # Load ROI mapping
        roi_mapping = load_roi_mapping(roi_mapping_file)

        # Add ROI info
        channel_to_network = roi_mapping.get('channel_to_network', {})
        channel_to_network_roi = roi_mapping.get('channel_to_network_roi', {})

        contrast_data['network'] = contrast_data['channel'].map(channel_to_network)
        contrast_data['network_roi'] = contrast_data['channel'].map(channel_to_network_roi)

        # Add subject
        contrast_data['subject'] = subject_std

        # Select columns
        channel_data = contrast_data[['subject', 'channel', 'effect', 'stat',
                                     'p_value', 'network', 'network_roi']].copy()
        channel_data.columns = ['subject', 'channel', 'activation', 't_statistic',
                               'p_value', 'network', 'network_roi']

        all_data.append(channel_data)

    if not all_data:
        raise ValueError(f"No session has HbO rows for {task} contrast '{contrast}'")

    result = pd.concat(all_data, ignore_index=True)

    # Cache result
    if cache is not None:
        cache[cache_key] = result

    return result


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def permutation_correlation(x, y, n_perm: int = 10000, *, rng=None) -> float:
    """Two-sided permutation p-value for Spearman rho.

    Returns the p-value alone. Not to be confused with
    ``utils.statistical_methods.permutation_correlation``, which returns
    ``(r, p, ci_lower, ci_upper)``.
    """
    rng = _resolve_rng(rng)
    observed = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    n_exceed = int((np.abs(null) >= abs(observed)).sum())
    return (1 + n_exceed) / (1 + n_perm)


def test_correlation(x: np.ndarray, y: np.ndarray, name: str,
                      n_perm: int = 10000, n_boot: int = 5000, *, rng=None) -> dict:
    """Test a single correlation with bootstrap CI and permutation p-value.

    Args:
        x, y: Data arrays
        name: Test name
        n_perm: Number of permutations for p-value
        n_boot: Number of bootstrap samples for CI
        rng: Generator for the permutation and bootstrap draws. Defaults to a
            generator on DEFAULT_SEED.

    Returns:
        dict with r, p, ci_lower, ci_upper, n, status
    """
    rng = _resolve_rng(rng)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 5:
        return {
            'r': np.nan, 'p': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'n': n, 'status': 'insufficient_data'
        }

    # Spearman correlation
    r, _ = stats.spearmanr(x, y)

    # A constant variable gives rho = NaN, and the permutation count would
    # then be 0, i.e. the smallest possible p. There is no correlation to test.
    if np.isnan(r):
        return {
            'r': np.nan, 'p': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'n': n, 'status': 'insufficient_data'
        }

    # Permutation p-value
    p_value = permutation_correlation(x, y, n_perm=n_perm, rng=rng)

    # Bootstrap CI
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r_boot, _ = stats.spearmanr(x[idx], y[idx])
        boot_rs.append(r_boot)

    ci_lower = np.percentile(boot_rs, 2.5)
    ci_upper = np.percentile(boot_rs, 97.5)

    return {
        'r': float(r), 'p': float(p_value),
        'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper),
        'n': n, 'status': 'completed'
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_forest_plot(results: dict, save_path: Path, level: str = 'roi'):
    """Create forest plot from results dict."""
    fig, ax = plt.subplots(figsize=(14 if level == 'channel' else 12, 10))

    # Collect tests
    all_tests = []
    for hypo_id, hypo_result in results.items():
        for test in hypo_result.get('tests', []):
            if test.get('status') == 'completed' and not np.isnan(test.get('r', np.nan)):
                all_tests.append({
                    'hypothesis': hypo_id,
                    'test_name': test.get('test_name', ''),
                    'var1': test.get('var1', '')[:20],
                    'var2': test.get('var2', '')[:20],
                    'r': test['r'],
                    'ci_lower': test['ci_lower'],
                    'ci_upper': test['ci_upper'],
                    'p': test.get('p', np.nan),
                    'p_fdr': test.get('p_fdr', np.nan),
                    'is_sig': test.get('significant_fdr', False)
                })

    if not all_tests:
        ax.text(0.5, 0.5, 'No results', ha='center', va='center')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Sort by correlation magnitude
    all_tests.sort(key=lambda x: abs(x['r']), reverse=True)

    # Limit to top 100 for channel-level (too many otherwise)
    if level == 'channel' and len(all_tests) > 100:
        all_tests = all_tests[:100]

    # Plot
    for i, test in enumerate(all_tests):
        y = len(all_tests) - 1 - i
        r = test['r']

        color = '#D9534F' if test['is_sig'] else '#2E86AB'
        ax.scatter(r, y, s=80 if level == 'channel' else 120,
                  color=color, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)
        ax.plot([test['ci_lower'], test['ci_upper']], [y, y],
               color=color, linewidth=1.5 if level == 'channel' else 2.5, alpha=0.7)

        # Label
        if level == 'roi':
            label = f"{test['hypothesis']}: {test['var1']} × {test['var2']}"
        else:
            label = f"{test['test_name'][:30]}"
        ax.text(-1.05 if level == 'roi' else -1.15, y, label,
               ha='right', va='center', fontsize=7 if level == 'channel' else 8)

    # Reference lines
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Spearman r with 95% CI', fontweight='bold')
    ax.set_yticks(range(len(all_tests)))
    ax.set_yticklabels([])
    ax.set_ylim(-1, len(all_tests))
    ax.set_xlim(-1.2, 1.2)
    ax.grid(True, alpha=0.3, axis='x')

    # Title
    n_tests = len(all_tests)
    ax.set_title(f'Effect Sizes ({level.upper()}-level, top {n_tests} shown)\n'
                f'Bootstrap 95% CI',
                fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results: dict, save_dir: Path, level: str, *, seed: int = None,
                  variant_name: str = None, h2_measure: str = None):
    """Save results to CSV and JSON.

    ``seed`` is written into results.json: the seed governs every p-value and
    CI in the file, so it has to travel with them rather than only reach the
    run log. ``variant_name`` and ``h2_measure`` (the resolved H2 behavioral
    measure) are likewise recorded when given, so the file names which of the
    run's analysis decisions produced it; omitted, results.json is unchanged.
    """
    # CSV
    rows = []
    for hypo_id, hypo_result in results.items():
        for test in hypo_result.get('tests', []):
            if test.get('status') == 'completed':
                rows.append({
                    'Hypothesis': hypo_id,
                    'Test': test.get('test_name', ''),
                    'r': test.get('r', np.nan),
                    'CI_lower': test.get('ci_lower', np.nan),
                    'CI_upper': test.get('ci_upper', np.nan),
                    'p': test.get('p', np.nan),
                    'p_fdr': test.get('p_fdr', np.nan),
                    'n': test.get('n', np.nan),
                    'Significant': 'Yes' if test.get('significant_fdr', False) else 'No'
                })

    if rows:
        pd.DataFrame(rows).to_csv(save_dir / 'results_summary.csv', index=False)

    # JSON - filter out non-serializable objects (DataFrames, etc.)
    def make_json_serializable(obj):
        """Recursively convert non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()
                    if k not in ('test_df', 'brief_col', 'behav_col', 'contrast')}
        if isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return None  # Skip DataFrames/Series
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NpEncoder, self).default(obj)

    json_results = make_json_serializable(results)
    output = {
        'level': level,
        'seed': seed,
        'n_tests': len(rows),
        'results': json_results
    }
    if variant_name is not None:
        output['variant'] = variant_name
    if h2_measure is not None:
        output['h2_measure'] = h2_measure

    with open(save_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)


def print_summary(results: dict):
    """Print summary of results."""
    n_sig = 0
    for hypo_result in results.values():
        for test in hypo_result.get('tests', []):
            if test.get('significant_fdr', False):
                n_sig += 1
                print(f"  {test.get('test_name', '')}: r={test.get('r', 0):.3f}, "
                      f"p_fdr={test.get('p_fdr', 1):.4f}")

    if n_sig == 0:
        print("  None")


def create_hypothesis_scatter_plots(hypo_id: str, hypo_def: dict,
                                     test_df: pd.DataFrame,
                                     brief_col: str, behav_col: str,
                                     contrast: str,
                                     results_dir: Path, level: str = 'roi',
                                     *, rng=None, tests=None, seed=None):
    """Create scatter plots for a hypothesis.

    Each panel prints the statistics of the test it illustrates:
    the matching entry of ``tests`` (``test_name`` ``<H>_self_brain``,
    ``<H>_brain_behav``, ``<H>_self_behav``) when given, else
    ``test_correlation`` with the confirmatory iteration counts on the
    generator the ROI-level test uses for that name and ``seed``. The
    channel-level plots show ROI means, so they print the ROI-level numbers.

    Creates 3 plots per hypothesis:
    1. Self-report × Brain (if brain data available)
    2. Brain × Behavior (if brain data available)
    3. Self-report × Behavior (always)

    For channel-level, uses the provided test_df (already filtered to relevant channels/ROIs).
    """
    from utils.statistical_methods import create_scatter_plot

    # The stats annotated on each panel are permutation-based, so the plots
    # need the same generator as the tests they illustrate.
    rng = _resolve_rng(rng)

    plot_dir = results_dir / "scatter_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_name = {t.get('test_name'): t for t in (tests or [])}

    def panel_stats(kind, x, y):
        name = f"{hypo_id}_{kind}"
        if name in by_name:
            return by_name[name]
        test_rng = rng_for_key(name, seed) if seed is not None else rng
        return test_correlation(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                                name, rng=test_rng)

    subjects = test_df['subject'].values
    hypo_name = hypo_def['name']

    # Plot 1: Self-report × Brain (if applicable)
    if 'activation' in test_df.columns:
        brain_label = f"Brain Activation ({contrast})" if contrast else "Brain Activation"
        if level == 'channel':
            # The channel-level scatter plots use the ROI-mean activation
            # (the same values as the ROI-level plots), not one channel.
            brain_label = f"Brain Activation ({contrast}, ROI mean)"

        fig = create_scatter_plot(
            test_df[brief_col].values,
            test_df['activation'].values,
            f"BRIEF-A {brief_col}",
            brain_label,
            subjects,
            f"{hypo_id}: {hypo_name} - Self-Report × Brain",
            plot_dir / f"{hypo_id}_self_brain.png",
            rng=rng,
            stats=panel_stats('self_brain', test_df[brief_col].values, test_df['activation'].values)
        )
        plt.close(fig)

    # Plot 2: Brain × Behavior (if applicable)
    if 'activation' in test_df.columns:
        fig = create_scatter_plot(
            test_df['activation'].values,
            test_df[behav_col].values,
            brain_label,
            behav_col,
            subjects,
            f"{hypo_id}: {hypo_name} - Brain × Behavior",
            plot_dir / f"{hypo_id}_brain_behav.png",
            rng=rng,
            stats=panel_stats('brain_behav', test_df['activation'].values, test_df[behav_col].values)
        )
        plt.close(fig)

    # Plot 3: Self-report × Behavior (always)
    fig = create_scatter_plot(
        test_df[brief_col].values,
        test_df[behav_col].values,
        f"BRIEF-A {brief_col}",
        behav_col,
        subjects,
        f"{hypo_id}: {hypo_name} - Self-Report × Behavior",
        plot_dir / f"{hypo_id}_self_behav.png",
        rng=rng,
        stats=panel_stats('self_behav', test_df[brief_col].values, test_df[behav_col].values)
    )
    plt.close(fig)

    print(f"  Saved scatter plots to {plot_dir}")
