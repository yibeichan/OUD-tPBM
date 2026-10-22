# Convergent Assessment of Executive Dysfunction in Opioid Use Disorder: Integrating Self-Report, Behavioral, and Neural Measures in a Transcranial Photobiomodulation Case Series

Yibei Chen<sup>1,†</sup>, Anayali Estudillo-Guerra<sup>2,†,*</sup>, Ine Mesia<sup>2</sup>, Paolo Cassano<sup>3</sup>

1. McGovern Institute for Brain Research, Massachusetts Institute of Technology, Cambridge, MA, United States
2. Hope House Addiction Services, Boston, MA, USA
3. Division of Neuropsychiatry, Massachusetts General Hospital, USA; Department of Psychiatry, Harvard Medical School, USA

† Yibei Chen and Anayali Estudillo-Guerra contributed equally and share first authorship.

\* Corresponding author: Anayali Estudillo-Guerra (aestudillo@hopehouse.org)

This repository holds the analysis code for the paper. It scores the clinical
assessments (BRIEF-A, PHQ-9, GAD-7, Brief Substance Craving Scale, treatment
expectancy) and the Stroop and Wisconsin Card Sorting Test (WCST) behavior,
fits a first-level GLM to each fNIRS session (Kernel Flow SNIRF files, MNE-NIRS), maps channels to
Schaefer 7-network ROIs per session, computes channel- and ROI-level group
statistics, tests the Unified Executive Function model (self-report, brain and
behavior measures of the same construct, H1-H5) at ROI level (confirmatory) and
channel level (exploratory), runs the longitudinal case series for the
participants with a second timepoint, and draws the manuscript figures. The
repository contains code only: no data, derived tables or rendered figures.

## Requirements and install

Python 3.11 or later. With [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/yibeichan/OUD-tPBM
cd OUD-tPBM
uv venv
uv pip install -e ".[dev,brain]"
source .venv/bin/activate
cp .env.example .env        # then set BASE_DIR and OUTPUT_DIR
```

The `brain` extra installs MNE-NIRS (with MNE) and h5py. It is needed by the
stages that read SNIRF files (`scripts/brain/`), by the two figure scripts that
build the montage from them (`fig1_glm_tmaps.py`,
`figS3_figS4_longitudinal_maps.py`), and by the test suite.
`dev` installs pytest.

The SLURM wrappers (`scripts/brain/*_glm_1st.sh`,
`scripts/slurm/variant_chain.sbatch`) activate a micromamba environment named
`fnirs`. They set no partition: export `SBATCH_PARTITION` to choose one, or
the cluster default applies. Their memory and time requests were sized for
the study data. The analyses for the paper were run in that environment
with these package versions (`uv.lock` resolves newer releases):

| Package | Version |
|---|---|
| Python | 3.13.2 |
| numpy | 2.2.3 |
| pandas | 2.2.3 |
| scipy | 1.15.1 |
| statsmodels | 0.14.5 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.3 |
| mne | 1.9.0 |
| mne-nirs | 0.7.1 |
| nilearn | 0.11.1 |

## Data

The data are available from the corresponding author upon reasonable request.

Every script reads two directories from `.env` (see `.env.example`):
`BASE_DIR`, which holds the input data, and `OUTPUT_DIR`, the run tree that
outputs are written to. `BASE_DIR` must have this layout:

```
BASE_DIR/data/
├── stroop_task_info.csv     # one row per session: filename, subj_id, task_id, task_order
├── wcst_task_info.csv
├── stroop/                  # SNIRF files and NeuroLume_{subject}_{task_id}_task_events.tsv
├── wcst/
├── assessments/             # {executive_function,phq9,gad7,craving,tea}_{timepoint}.csv
└── atlas/                   # flow2_absorption_Schaefer2018_100Parcels_7Networks.npz,
                             # Schaefer2018_100Parcels_7Networks_order.txt
```

`task_order` is 1 for baseline, 2 for post and 3 for follow-up. Subjects are
coded `S1` to `S12`. `scripts/check_public_hygiene.py` fails on a tracked
file that holds the raw participant-ID prefix, an absolute path under the
site's storage roots, under the macOS `Users` root or under `/home/<user>`, an email address other than the
corresponding author's address above, or prose that names an AI assistant
(or addresses one) in the documentation; when `BASE_DIR` in `.env` points at the data, it also
fails on any acquisition session ID listed in the task-info files. The site's
storage roots and the identifier patterns are not tracked: they come from
`HYGIENE_SITE_ROOTS` and `HYGIENE_ID_PATTERNS` in `.env` (see `.env.example`),
and without them the check runs with generic defaults only. The run before
publishing uses `python scripts/check_public_hygiene.py --require-settings`,
which fails when either setting is missing.

## Running the pipeline

Run from the repository root. Each command reads `BASE_DIR` and `OUTPUT_DIR`
from `.env`; a variable set in the environment takes precedence.

### Shared stages

Stages 1-4 do not depend on the analysis variant. They write into one source
tree (`OUTPUT_DIR` set to it) and are run once.

| # | Stage | Command | Writes |
|---|---|---|---|
| 1 | Assessment scores | `python scripts/assessments/00_load_assessment_data.py`, then `01_analyze_executive_function.py`, `02_analyze_phq9.py`, `03_analyze_gad7.py`, `04_analyze_craving.py`, `05_analyze_tea.py` in `scripts/assessments/` | `01_assessments/` |
| 2 | Behavior | `python scripts/behav/01_stroop_behav.py` and `python scripts/behav/01_wcst_behav.py` | `02_behav/` |
| 3 | First-level GLM (SLURM array, one task per session) | `bash scripts/brain/stroop_glm_1st.sh` and `bash scripts/brain/wcst_glm_1st.sh` (each submits its own array job) | `03_glm/` |
| 4 | Channel mappings | `python scripts/brain/step2a_glm_to_parcellation.py`, then `python scripts/brain/step2b_parcellation_to_roi.py` | `04_channel_mappings/` |

`scripts/assessments/01_plot_executive_function_t-score.py` and
`06_longitudinal_comparison.py` draw and tabulate the scores; no later stage
reads their output.

### Variant run trees

Each analysis variant gets its own run tree, made by
`scripts/setup_run_tree.py`:

```bash
python scripts/setup_run_tree.py --variant main --source <source tree> --dest <run tree>
```

It copies `config/analysis_variants/<variant>.json` into the tree as
`analysis_variant.json` and symlinks stages 1-4 from the source tree. Stages
from `step3a` onward read `analysis_variant.json` through
`scripts/analysis_variant.py` and refuse to run when it is missing or has a
missing, unknown or invalid entry. Build the `main` tree first, in a
directory named `main` (`MAIN_TREE_NAME` in `setup_run_tree.py`), and run its `2c` stage before
creating any other variant's tree: the other trees are made next to it and
link its `04b_session_info/`. `supp_count` and `supp_brain_activation` change
only `09_hypothesis_testing/` and `10_longitudinal/`, so their trees can link
`main`'s `05_` to `08_` directories instead of recomputing them
(`--link-from <main tree> --link-stages 05 06 07 08`); `supp_count` can also
link `10_longitudinal/roi_activations_alltimepoints/` (add `10`).

To keep finished run trees from being overwritten, list path fragments that
mark them in `PROTECTED_TREE_MARKERS` in `.env` (comma-separated; see
`.env.example`). Every stage that writes into `OUTPUT_DIR` (the assessment
and behavior scripts, every `scripts/brain/step*` stage, the hypothesis
tests, the longitudinal scripts, `variant_chain.sbatch` and the figure
scripts) refuses an `OUTPUT_DIR` that contains one. Unset, nothing is
protected.

The remaining stages run as one SLURM job per tree:

```bash
RUN_TREE=/path/to/run_tree STAGES="2c 3a 3b 3c 4 roi10 hypo_roi hypo_channel long long_expl" \
  sbatch scripts/slurm/variant_chain.sbatch
```

`STAGES` takes any subset of these, in this order. The seed is the one the
paper used; `scripts/seeding.py` derives a separate generator for each test
from it.

| Stage | Command (as run by `variant_chain.sbatch`) | Writes |
|---|---|---|
| `2c` | `python scripts/brain/step2c_session_info.py` | `04b_session_info/` |
| `3a` | `python scripts/brain/step3a_channel_group_analysis.py --task {wcst,stroop} --seed 20260922` | `05_channel_group_analysis/` |
| `3b` | `python scripts/brain/step3b_roi_extraction.py --task {wcst,stroop} --baseline-only` | `06_roi_activation/` |
| `3c` | `python scripts/brain/step3c_roi_group_analysis.py --task {wcst,stroop} --seed 20260922` | `07_roi_analysis/` |
| `4` | `python scripts/brain/step4_glm_topoplots.py --task {wcst,stroop}` | `08_glm_topoplots/` |
| `roi10` | `python scripts/longitudinal/extract_roi_alltimepoints.py --task both` | `10_longitudinal/roi_activations_alltimepoints/` |
| `hypo_roi` | `python scripts/hypo_test/unified_ef_model_roi.py --seed 20260922` | `09_hypothesis_testing/unified_ef_model_roi/` |
| `hypo_channel` | `python scripts/hypo_test/unified_ef_model_channel.py --seed 20260922` | `09_hypothesis_testing/unified_ef_model_channel/` |
| `long` | `python scripts/longitudinal/unified_ef_longitudinal.py --seed 20260922` | `10_longitudinal/unified_ef/` |
| `long_expl` | `python scripts/longitudinal/longitudinal_exploratory.py` | `10_longitudinal/` (top-level tables and `figures/`) |

### Figures

The figure scripts read a finished `main` run tree and refuse a tree of any
other variant, or one that lacks an input they need (the message lists the
missing files). Each writes one file per panel, for assembly into the
composite figure, under `$OUTPUT_DIR/figures/`:

```bash
OUTPUT_DIR=/path/to/main_run_tree python scripts/figures/fig1_glm_tmaps.py
OUTPUT_DIR=/path/to/main_run_tree python scripts/figures/fig2_figS1_self_report.py
OUTPUT_DIR=/path/to/main_run_tree python scripts/figures/fig3_figS2_behavior.py
OUTPUT_DIR=/path/to/main_run_tree python scripts/figures/figS3_figS4_longitudinal_maps.py
```

`scripts/brain/README.md`, `scripts/hypo_test/README.md` and
`scripts/longitudinal/README.md` describe the stages in more detail.

## Which code produces what

### Analysis variants

`main` is the primary analysis and backs the main text. Each `supp_*` variant
changes one decision of `main` and backs a supplementary analysis. The
decisions are the keys of `config/analysis_variants/*.json`, checked by
`scripts/analysis_variant.py`.

| Variant | Decision | Setting |
|---|---|---|
| `main` | all | Long channels only (10 mm or more; short channels are nuisance regressors in the first-level GLM and are then dropped). Task sessions at or below chance accuracy (Stroop 1/3, WCST 1/4) are excluded. The H2 behavioral measure is the WCST perseverative error rate. BRIEF-A administrations that fail a validity scale are excluded. Each participant's second timepoint is the post or follow-up visit with more usable levels (a tie goes to post). The WCST contrasts that use the regular-incorrect regressor (`correct_vs_incorrect`, `change_vs_regular`, `learning`) are treated as missing for sessions with no regular-incorrect trials. In the longitudinal analysis, lower brain activation at the second timepoint counts as improvement. |
| `supp_all_channels` | `channels` | Short channels are kept in every stage after the first-level GLM. |
| `supp_low_accuracy` | `exclude_low_accuracy` | Sessions at or below chance accuracy are kept. |
| `supp_count` | `h2_metric` | The H2 behavioral measure is the perseverative error count instead of the rate. |
| `supp_brain_activation` | `brain_direction` | In the longitudinal analysis, higher brain activation counts as improvement. |

### Outputs

Every variant's run tree holds all of these folders: stages 1-4 and
`04b_session_info/` are shared through symlinks, and the rest are computed per
variant or linked from `main` as described above. The main text uses the
`main` tree; the supplement uses the `supp_*` trees and the S-figures, which
are drawn from the `main` tree. Which variant's tree a folder comes from
decides what it feeds: the `main` tree feeds the main text, and a `supp_*` tree
feeds a supplementary analysis. No figure script reads `08_glm_topoplots/`.

| Output folder (under the run tree) | Script | Read by the figure scripts |
|---|---|---|
| `01_assessments/` | `scripts/assessments/00`-`05_*.py` | Figures 2, S1 |
| `02_behav/` | `scripts/behav/01_{stroop,wcst}_behav.py` | No (read by `2c`, `hypo_*`, `long` and `long_expl`) |
| `03_glm/` | `scripts/brain/step1_glm_{stroop,wcst}.py` | Figures S3, S4 |
| `04_channel_mappings/` | `scripts/brain/step2a_*.py`, `step2b_*.py` | No (read by `3a`, `3b`, `roi10` and `hypo_channel`) |
| `04b_session_info/` | `scripts/brain/step2c_session_info.py` | Figures S3, S4 |
| `05_channel_group_analysis/` | `scripts/brain/step3a_channel_group_analysis.py` | Figure 1 |
| `06_roi_activation/` | `scripts/brain/step3b_roi_extraction.py` | No (read by `3c` and `hypo_roi`) |
| `07_roi_analysis/` | `scripts/brain/step3c_roi_group_analysis.py` | No (read by `4`) |
| `08_glm_topoplots/` | `scripts/brain/step4_glm_topoplots.py` | No |
| `09_hypothesis_testing/` | `scripts/hypo_test/unified_ef_model_{roi,channel}.py` | No |
| `10_longitudinal/` | `scripts/longitudinal/extract_roi_alltimepoints.py`, `unified_ef_longitudinal.py`, `longitudinal_exploratory.py` | Figures 2, 3, S1, S2, S3, S4 |

| Figure | Script | Output folder (under `$OUTPUT_DIR/figures/`) | Feeds |
|---|---|---|---|
| Figure 1: channel-level group t maps, Stroop and WCST contrasts | `scripts/figures/fig1_glm_tmaps.py` | `main/Figure1_Stroop/`, `main/Figure1_WCST/` | Main text |
| Figure 2: self-report trajectories (BRIEF-A GEC, PHQ-9, GAD-7, BSCS) | `scripts/figures/fig2_figS1_self_report.py` | `main/Figure2_self_report/` | Main text |
| Figure S1: BRIEF-A Inhibit, Shift and Working Memory trajectories | `scripts/figures/fig2_figS1_self_report.py` | `supplementary/FigureS1_BRIEF-A_subscales/` | Supplement |
| Figure 3: Stroop and WCST behavior, baseline to second timepoint | `scripts/figures/fig3_figS2_behavior.py` | `main/Figure3_behavior/` | Main text |
| Figure S2: behavior change heatmaps | `scripts/figures/fig3_figS2_behavior.py` | `supplementary/FigureS2_behavior_change_heatmaps/` | Supplement |
| Figure S3: pre and post β maps, the four contrasts the H1-H4 pathways test | `scripts/figures/figS3_figS4_longitudinal_maps.py` | `supplementary/FigureS3_longitudinal_maps_pathway_contrasts/` | Supplement |
| Figure S4: pre and post β maps, the other three contrasts | `scripts/figures/figS3_figS4_longitudinal_maps.py` | `supplementary/FigureS4_longitudinal_maps_other_contrasts/` | Supplement |

Figures 1 and S3-S4 also read the SNIRF files under `BASE_DIR/data/` for the
montage.

Panel file names keep the prefixes of the panel files the composite figures
were assembled from, so they do not follow the figure numbers: Figure 1's
Stroop panels are `fig1_*` and its WCST panels `fig2_*`, Figure 2's panels are
`fig6_*`, Figure 3's are `fig5_*`, Figures S3 and S4 share `fig4_*`, and
Figures S1 and S2 use `figS_brief_subscales_*` and
`figS_behavior_change_heatmap_*`.

## Tests

```bash
python scripts/check_public_hygiene.py
pytest tests/ -q
```

The unit tests need no data. The study-data checks are skipped unless these
are set:

| Setting | What it enables |
|---|---|
| `OUD_TPBM_BASE_DIR` (environment) | Checks against the raw inputs; set it to the `BASE_DIR` holding `data/`. |
| `OUD_TPBM_REFERENCE_TREE` (environment) | Checks against a finished run tree, e.g. a `main` tree. |
| `tests/local_pinned_values.json` (untracked file) | Expected per-participant values for those checks. `tests/local_pinned_values.json.example` shows its layout; a check whose key is missing is skipped. |

The data-gated tests read only `OUD_TPBM_BASE_DIR` and
`OUD_TPBM_REFERENCE_TREE`, never `BASE_DIR` or `OUTPUT_DIR`, so a lab `.env`
does not switch them on by accident.

## License

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the paper. `CITATION.cff` holds the
software citation metadata.
