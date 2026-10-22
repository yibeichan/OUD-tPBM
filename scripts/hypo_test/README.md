# Unified EF Model - Executive Function Analysis Framework

**Location:** `scripts/hypo_test/unified_ef_model_*.py`

## Overview

The Unified EF Model integrates three data types to test executive function:

```
Self-Report (BRIEF-A)  →  Brain (fNIRS)  →  Behavior (Task Performance)
       (Trait)               (State)              (Outcome)
```

This is a **convergent validity framework**, NOT formal mediation. We test whether trait EF (BRIEF-A), neural implementation (fNIRS), and behavioral outcome (task performance) align as predicted by theory.

## Theoretical Model

| Hypothesis | Construct | BRIEF-A | Brain | Behavior |
|------------|-----------|---------|-------|----------|
| **H1** | Inhibition | Inhibit | SalVentAttn/Cont (Stroop effect) | Interference |
| **H2** | Cognitive Flexibility | Shift | Cont (WCST change vs regular) | Perseverative error rate (count in the `supp_count` variant) |
| **H3** | Rule Learning | Working Memory | Cont (WCST learning) | WM maintenance decline |
| **H4** | Attentional Stability | Working Memory | Cont (Stroop mixing cost) | RT variability |
| **H5** | Global EF | GEC | - | Global performance z-score |

`PRIMARY_HYPOTHESES` in `unified_ef_model_common.py` is the single definition of this table (target ROIs included). Its H2 behavioral measure is the module default, `perseverative_errors`; `main()` replaces it with the run tree's `h2_metric` (`resolve_hypotheses`), which is the rate in the `main` variant. H5's behavior is the mean of the z-scored Stroop overall accuracy and the z-scored proportion of WCST rule periods mastered.

## File Structure

```
scripts/hypo_test/
├── unified_ef_model_common.py      # Hypothesis definitions, data loading, correlation test, plots, result writers
├── unified_ef_model_roi.py         # ROI-level analysis (confirmatory)
├── unified_ef_model_channel.py     # Channel-level analysis (exploratory)
└── utils/
    └── statistical_methods.py      # Behavior and assessment loaders, plotting setup, scatter helper
```

## Inputs

Both scripts read a pipeline run tree. They need `BASE_DIR` and `OUTPUT_DIR` (in `.env` or the environment) and an `analysis_variant.json` at the `OUTPUT_DIR` root (created by `scripts/setup_run_tree.py`). From the tree they read:

- `01_assessments/` BRIEF-A T-scores (baseline), with invalid administrations blanked when the variant says so
- `02_behav/` per-session behavior results
- `06_roi_activation/{task}/all_subjects_roi_activation_network_roi.csv` (ROI scope), exactly one baseline session per subject
- `03_glm/` and `04_channel_mappings/` (channel scope), long channels only unless the variant keeps all channels
- `$BASE_DIR/data/{stroop,wcst}_task_info.csv`

## Analysis Levels

### ROI-Level (Confirmatory)
- **Data:** Pre-aggregated; the unweighted mean of the matched target ROIs' per-subject means
- **Tests:** per H1-H4, self-report × brain and brain × behavior (the 8 primary tests), plus self-report × behavior; H5 is one direct test
- **FDR:** Benjamini-Hochberg across the 8 primary tests only; `self_behav` and H5 are reported but not corrected

### Channel-Level (Exploratory)
- **Data:** Raw per-channel values, no aggregation, restricted to the hypothesis's target ROIs
- **Tests:** per channel, self-report × brain and brain × behavior; channels with fewer than five participants are skipped
- **FDR:** Benjamini-Hochberg within each hypothesis, pooling both sub-pathways across all its tested channels

## Statistical Methods

- **Correlation:** Spearman rank correlation
- **Permutation p-value:** (1 + count) / (1 + n_perm); 10,000 permutations at ROI level and 50,000 at channel level. The channel count is set so that the smallest achievable p is below the FDR threshold of a large channel family
- **Confidence interval:** percentile bootstrap 95% CI; 5,000 resamples at ROI level, 1,000 at channel level
- **Seeding:** `--seed` (default in `scripts/seeding.py`) seeds a generator per test from the test's name, so a test's p-value does not depend on which tests ran before it
- **Sample:** baseline sessions only (`task_order=1`); each test reports its own n

## Results

The results are written to the run tree's `09_hypothesis_testing/*/results_summary.csv` and `results.json`.

## Usage

```bash
python scripts/hypo_test/unified_ef_model_roi.py --seed 20260922
python scripts/hypo_test/unified_ef_model_channel.py --seed 20260922
```

The seed shown is the one `scripts/slurm/variant_chain.sbatch` passes.

## Output Structure

```
OUTPUT_DIR/09_hypothesis_testing/
├── unified_ef_model_roi/
│   ├── forest_plot.png          # Effect sizes with 95% CI
│   ├── results_summary.csv      # All results
│   ├── results.json             # Full results with metadata
│   ├── merged_data.csv          # Input dataset
│   └── scatter_plots/           # Per-hypothesis scatter plots
└── unified_ef_model_channel/
    ├── forest_plot.png          # Top 100 channels shown
    ├── results_summary.csv
    ├── results.json
    ├── merged_data.csv
    └── scatter_plots/
```
