# Longitudinal Analysis Scripts

This directory contains the scripts for the longitudinal case series: the participants with a second timepoint, across self-report, neural activation (fNIRS), and behavioral performance.

These scripts read a pipeline run tree under `$OUTPUT_DIR` and write into its `10_longitudinal/`. Figures 2, 3 and S1-S4 read their outputs (see the top-level README, "Which code produces what").

---

## Scripts

### 0. ROI extraction at every timepoint

**Script**: `extract_roi_alltimepoints.py` (stage `roi10` in `scripts/slurm/variant_chain.sbatch`)

The cross-sectional pipeline writes baseline-only ROI activations. This script reuses step 3b's per-subject extraction for every session, with the contrasts the convergence analysis needs, and writes `OUTPUT_DIR/10_longitudinal/roi_activations_alltimepoints/{task}/all_subjects_roi_activation_network_roi.csv`. Both other scripts read it.

```bash
python scripts/longitudinal/extract_roi_alltimepoints.py --task both
```

### 1. Hypothesis-driven analysis

**Script**: `unified_ef_longitudinal.py` (stage `long`)

**Purpose**: Tests convergent improvement across self-report, neural, and behavioral levels for the five hypotheses (H1-H5) of the unified EF model.

**Method**: A Convergence Index (directional consistency across the levels) alone sets each participant × hypothesis classification (`convergent_improvement`, `partial_convergence`, `mixed`, `partial_divergence`, `divergent`). A Reliable Change Index with a small-sample threshold is written to the outputs but not used to classify or report. The expected direction of the brain change comes from the run's variant (`brain_direction`: lower activation counts as improvement in `main`, higher in `supp_brain_activation`), and H2's behavior is the variant's `h2_metric`.

**Output**: `OUTPUT_DIR/10_longitudinal/unified_ef/` (see below)

```bash
python scripts/longitudinal/unified_ef_longitudinal.py --seed 20260922
```

### 2. Exploratory analysis

**Script**: `longitudinal_exploratory.py` (stage `long_expl`)

**Purpose**: Exploratory case-series tables and figures for all available behavioral, neural, and assessment metrics, with change scores and the share of participants who improved (no effect sizes). It also writes the long-format files `assessments_long.csv` and `behavior_long.csv`, and `consistency_summary.csv`; Figures 2, 3, S1 and S2 read `behavior_long.csv` and `consistency_summary.csv`.

**Output**: `OUTPUT_DIR/10_longitudinal/` (root-level CSVs and `figures/`)

```bash
python scripts/longitudinal/longitudinal_exploratory.py
```

### 3. Choosing the second timepoint

**Module**: `timepoint_selection.py`, used by both analysis scripts.

A participant's second timepoint is their post (task_order 2) or follow-up (task_order 3) visit, whichever has more usable measurement levels (BRIEF-A self-report, Stroop session, WCST session). A tie goes to post. Excluded task sessions and BRIEF-A administrations that fail the validity check do not count as usable levels. The chosen visit is relabelled order 2 / "post" in the analysis; the written files keep the real visit in `source_task_order` and `source_timepoint`. `unified_ef/second_timepoint_selection.csv` records each participant's choice, the number of levels and which levels.

Questionnaire pairing (`apply_second_timepoint_questionnaires`, exploratory script only): when the chosen visit has no PHQ-9 or GAD-7 administration but another post-baseline visit has one, that questionnaire's second-timepoint value comes from the post-baseline visit closest to the chosen one (the earlier visit on a tie), and its `source_task_order` names that visit. Each questionnaire is paired on its own. BRIEF-A is never substituted. These questionnaires are descriptive and enter no selection, hypothesis test or Convergence Index. This stage does not load the BSCS; `scripts/figures/fig2_figS1_self_report.py` pairs the BSCS by the same rule for Figure 2, so `consistency_summary.csv` has no BSCS row.

---

## Comparison

| Aspect | `unified_ef_longitudinal.py` | `longitudinal_exploratory.py` |
|--------|------------------------------|-------------------------------|
| **Purpose** | Test H1-H5 convergence | Explore all metrics |
| **Scope** | 5 hypotheses | All behavioral + brain + assessment |
| **Methods** | Convergence Index (RCI written out, not used) | Descriptive stats, change scores |
| **Output** | Classifications, heatmap, trajectories | Trajectory plots, change tables, long-format files |

---

## Requirements

- The `fnirs` environment (or `.[brain]` extras plus the base dependencies)
- `BASE_DIR` and `OUTPUT_DIR` set (via `.env`), and an `analysis_variant.json` at the `OUTPUT_DIR` root
- Under `OUTPUT_DIR`: `02_behav/{task}/` per-session behavior results, `01_assessments/` scores and T-scores, `03_glm/{task}/` GLM contrasts and, for the convergence analysis, the output of `extract_roi_alltimepoints.py`
- `$BASE_DIR/data/{stroop,wcst}_task_info.csv`

---

## Timepoints

- **baseline** (task_order=1)
- **post** (task_order=2)
- **followup** (task_order=3), where recorded

Which participants have a second timepoint, and which visit each one uses, is not fixed in this README: it is computed from the data and recorded in `unified_ef/second_timepoint_selection.csv`.

---

## Theoretical Framework

The longitudinal analysis extends the **Unified Executive Function Model** (see `scripts/hypo_test/README.md`).

### Core Premise

If executive function is a coherent construct, then when EF changes (e.g., due to treatment), we should observe **convergent changes** across:

1. **Self-Report** (BRIEF-A): Perceived difficulties in daily life
2. **Neural** (fNIRS): Task-evoked brain activation
3. **Behavior** (Task performance): Observable cognitive outcomes

### Parallel Processes Model

Changes at different levels may occur at different rates, but should be in the **same direction** for true EF improvement. With two timepoints the order of changes across levels cannot be tested.

---

## Output Directory Structure

```
${OUTPUT_DIR}/10_longitudinal/
│
│  # From extract_roi_alltimepoints.py:
├── roi_activations_alltimepoints/{task}/all_subjects_roi_activation_network_roi.csv
│
│  # From longitudinal_exploratory.py:
├── assessment_changes.csv
├── assessments_long.csv
├── behavior_changes.csv
├── behavior_long.csv
├── behavior_wide.csv
├── brain_roi_activations.csv
├── brain_roi_changes.csv
├── consistency_summary.csv
├── analysis_summary.json
├── figures/
│
│  # From unified_ef_longitudinal.py:
└── unified_ef/
    ├── second_timepoint_selection.csv
    ├── hypothesis_convergence.csv
    ├── subject_changes.csv
    ├── subject_convergence_classifications.csv
    ├── convergence_summary.csv
    ├── analysis_summary.json
    └── figures/        # convergence_heatmap.png, subject_*_trajectories.png, hypothesis_*_convergence.png
```
