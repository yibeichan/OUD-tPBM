# Brain Scripts Organization

This directory contains the fNIRS analysis stages: first-level GLM, channel-to-ROI mapping, and group statistics. They need Kernel Flow SNIRF inputs (not shipped) and, from step 3a onward, a run tree with an `analysis_variant.json` (see the top-level README, "Running the pipeline").

## Design decision: per-session mapping

Each session records its own set of HbO channels, so a channel index such as `S1_D1` does not mark the same scalp position in every session unless the channels are named by optode position. Two things follow:

- Channels are named by optode position in the full probe's label order (`channel_names.py`, `config/flow_probe_template.json`). S4's probe lacks four optodes, so MNE's index-based names would point some of its channels at other optodes; never pool channels across sessions by raw MNE names.
- The channel-to-ROI mapping is built per session from that session's SNIRF file: step 2a writes the GLM→parcellation mapping and step 2b the parcellation→ROI mapping. Later stages load the mapping for the session they are reading.

## Analysis pipeline

```
Step 1   step1_glm_stroop.py / step1_glm_wcst.py       first-level GLM per session
Step 2a  step2a_glm_to_parcellation.py                 per-session GLM → parcellation mapping
Step 2b  step2b_parcellation_to_roi.py                 per-session parcellation → ROI mapping
Step 2c  step2c_session_info.py                        per-session short channels, WCST condition counts
Step 3a  step3a_channel_group_analysis.py              channel-level group statistics
Step 3b  step3b_roi_extraction.py                      ROI activation per subject
Step 3c  step3c_roi_group_analysis.py                  ROI-level group statistics
Step 4   step4_glm_topoplots.py                        topoplots from the group statistics
```

Steps 3a and 3b/3c are independent of each other; both read the step 1 output and the step 2 mappings. Brain-behavior hypothesis testing is in `scripts/hypo_test/` (see `hypo_test/README.md`).

## Script descriptions

### Step 1: first-level GLM
- `step1_glm_stroop.py`, `step1_glm_wcst.py`: subject-level GLM per session. The short-separation channels enter the design matrix as nuisance regressors; the noise model is AR(1) with an ordinary-least-squares fallback that is logged per session. The WCST script adds the mean-centred trial duration as a parametric modulator of the regular-correct regressor.
- `stroop_glm_1st.sh`, `wcst_glm_1st.sh`: SLURM array submission scripts, one task per row of `${BASE_DIR}/data/{stroop,wcst}_task_info.csv`.
- `trial_onsets.py`, `channel_names.py`, `channel_quality.py`: shared helpers (onsets from the SNIRF stims, canonical channel names, flat or non-finite channel screen).

### Step 2: per-session channel mapping
- `step2a_glm_to_parcellation.py`: reads each session's SNIRF file and writes one `mapping_{task}_{subject}_{session}.json` plus `session_index.json` under `04_channel_mappings/`. Per-session output is the default. The other flags (`--validate-all`, `--match-sessions`, `--check-consistency`, `--verbose`, `--snirf-file`) are diagnostics that write no mapping files.
- `step2b_parcellation_to_roi.py`: maps each session's parcels to Schaefer ROIs (100-parcel, 7-network atlas; Schaefer et al., 2018) and writes `roi_mapping_{task}_{subject}_{session}.json`.
- `step2c_session_info.py`: per-session short-channel lists (distance under 10 mm) and WCST condition counts, written to `04b_session_info/`. Later stages use them to drop short channels and to mask WCST maintenance contrasts for sessions with no regular-incorrect trials.

### Step 3a: channel-level group analysis
- `step3a_channel_group_analysis.py`: group statistics at channel level.
  - One-sample sign-flipping permutation test per channel (H0: activation = 0; 10,000 permutations)
  - Cohen's d with BCa bootstrap 95% CI (5,000 resamples)
  - FDR (Benjamini-Hochberg) within each contrast
  - A channel is tested only when every included baseline session has a value for it, so the number of subjects per channel is in the `n_subjects` column
  - Output: `05_channel_group_analysis/{task}/{task}_{contrast}_channel_statistics.csv`, the combined table and a summary JSON

### Step 3b: ROI extraction
- `step3b_roi_extraction.py`: loads each session's GLM contrasts with its own ROI mapping and aggregates channels to ROIs by the unweighted mean across channels (`--method median` is for inspection). Baseline sessions only by default (`--all-sessions` is for inspection and is not a valid input to later stages). Output: per-session ROI files and `all_subjects_roi_activation_network_roi.csv` under `06_roi_activation/{task}/`.

### Step 3c: ROI-level group statistics
- `step3c_roi_group_analysis.py`: one-sample t-test of ROI activation against zero, group means, bootstrap CIs and FDR, for every contrast. Output: `07_roi_analysis/{task}/group_roi_statistics_network_roi.csv`, a run-metadata JSON and a summary figure. Brain-behavior correlations are not computed here; they are in `scripts/hypo_test/`.

### Step 4: topoplots
- `step4_glm_topoplots.py`: topoplots from the step 3a and 3c statistics. The FDR maps paint only the channels or ROIs with q < 0.05; the uncorrected maps show every value. Output: `08_glm_topoplots/{task}/`.

### Utilities
- `roi_utils.py`: loads GLM files together with their session-specific ROI mapping (`get_glm_with_roi_mappings()`, `load_roi_mapping()`), and reads the run tree's variant and exclusions.
- `session_info.py`: reads the step 2c session tables.

## Quick start

Run from the repository root, with `BASE_DIR` and `OUTPUT_DIR` in `.env` and the `fnirs` environment (or `uv pip install -e ".[brain]"`) active.

```bash
# 1. First-level GLM (each launcher submits its own SLURM array job)
bash scripts/brain/stroop_glm_1st.sh
bash scripts/brain/wcst_glm_1st.sh

# 2. Per-session mappings
python scripts/brain/step2a_glm_to_parcellation.py
python scripts/brain/step2b_parcellation_to_roi.py
python scripts/brain/step2c_session_info.py

# 3. Group statistics
python scripts/brain/step3a_channel_group_analysis.py --task stroop
python scripts/brain/step3a_channel_group_analysis.py --task wcst
python scripts/brain/step3b_roi_extraction.py --task stroop
python scripts/brain/step3b_roi_extraction.py --task wcst
python scripts/brain/step3c_roi_group_analysis.py --task stroop
python scripts/brain/step3c_roi_group_analysis.py --task wcst

# 4. Topoplots
python scripts/brain/step4_glm_topoplots.py --task stroop
python scripts/brain/step4_glm_topoplots.py --task wcst
```

The stage chain, with the seeds used for the paper, is `scripts/slurm/variant_chain.sbatch`.

Every stage here refuses an `OUTPUT_DIR` that contains a marker listed in
`PROTECTED_TREE_MARKERS` (see `.env.example`).

## Contrasts

### Stroop
- `stroop_effect`: incongruent − congruent_mixed (primary)
- `mixing_cost`: congruent_mixed − congruent_pure
- `incongruent_vs_pure`: incongruent − congruent_pure

### WCST
- `correct_vs_incorrect`: regular_correct − regular_incorrect
- `change_vs_regular`: change − mean(regular_correct, regular_incorrect)
- `learning`: post_change − mean(regular_correct, regular_incorrect)
- `duration_effect_correct`: the trial-duration modulator of regular_correct

A session with no regular-incorrect trials has no `regular_incorrect` regressor; its `correct_vs_incorrect`, `change_vs_regular` and `learning` values are still written to `03_glm/`. When the variant sets `mask_wcst_without_regular_incorrect`, `roi_utils.read_glm_contrasts` drops those rows for such a session, and the ROI and hypothesis stages treat them as missing. The channel-level group test (`step3a`) needs every included baseline session to have each contrast, so it stops with an error if a masked session is among them rather than testing fewer sessions.

## Output structure

```
OUTPUT_DIR/
├── 03_glm/{task}/
│   ├── {subject}_{session}_glm.pkl
│   ├── {subject}_{session}_glm_contrasts.csv
│   └── {subject}_{session}_glm_session.json
├── 04_channel_mappings/
│   ├── mapping_{task}_{subject}_{session}.json        # step 2a
│   ├── roi_mapping_{task}_{subject}_{session}.json    # step 2b
│   └── session_index.json
├── 04b_session_info/                                  # step 2c
├── 05_channel_group_analysis/{task}/                  # step 3a
├── 06_roi_activation/{task}/                          # step 3b
├── 07_roi_analysis/{task}/                            # step 3c
└── 08_glm_topoplots/{task}/                           # step 4
```

## ROI organization (Schaefer atlas)

Networks: Vis, SomMot, DorsAttn, SalVentAttn, Cont (frontal-parietal control), Default, and Limbic. Limbic is excluded from the analyses because fNIRS cannot sample it. ROIs are sub-regional labels of the form `{network}_{region}` (for example `Cont_PFCl`, `SalVentAttn_PFCl`); the analyses use the `network_roi` level.

## Notes

- **Environment**: the analyses for the paper ran in the `fnirs` micromamba environment.
- **Paths**: `BASE_DIR` and `OUTPUT_DIR` come from `.env`; nothing is hard-coded.
