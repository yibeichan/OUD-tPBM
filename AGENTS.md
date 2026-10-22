# Contributor guide

## Project status

This repository is the analysis code for a paper. The analysis is complete,
so changes should be small and additive (typos, broken links, bug fixes,
follow-on analyses) rather than widening the scope. Open an issue before a
large refactor.

## Sample

- n = 12 at baseline. A participant can still lack a level (a session excluded
  by the analysis rules, a BRIEF-A administration that fails its validity
  check), so each test reports its own n.
- n = 5 with a second timepoint. A participant's second timepoint is their post
  or follow-up visit, whichever has more usable measurement levels (a tie goes
  to post); a follow-up chosen this way is labelled "post" in the analysis, and
  the longitudinal output files name the real visit in `source_timepoint`
  (`scripts/longitudinal/timepoint_selection.py`).

## Environment

```bash
# First-time setup
uv venv
uv pip install -e ".[dev,brain]"

# Daily use
source .venv/bin/activate
```

The `brain` extra (MNE-NIRS and h5py) is needed by the stages that read SNIRF files, by the figure scripts that build the montage, and by the test suite.

Optional extra:
- `uv pip install -e ".[bayes]"`: pymc and arviz for the Bayesian comparators in `scripts/hypo_test/utils/`

The analyses for the paper ran in a micromamba environment named `fnirs` (its
main package versions are in the README), not in the uv environment. `uv.lock`
resolves newer releases, so treat it as the test environment, not as the
record of what computed the outputs. The SLURM wrappers activate `fnirs`.

## Build, test and development commands

- Tests: `pytest tests/` (unit tests, plus study-data checks that are skipped unless `OUD_TPBM_BASE_DIR` or `OUD_TPBM_REFERENCE_TREE` is set and `tests/local_pinned_values.json` is present; see "Tests" in the README)
- Content check: `python scripts/check_public_hygiene.py` (standard library only; run it before the tests)
- Pipeline stages and figure scripts: see "Running the pipeline" in the README

## Coding style

- PEP 8, 4-space indent, snake_case, type hints in new code
- pathlib for filesystem access; f-strings for formatting
- Config via `python-dotenv` (`BASE_DIR`, `OUTPUT_DIR`); no hard-coded paths
- Tests use `tmp_path` + `monkeypatch.setenv` for filesystem fixtures

## Subject identifiers

- Subjects are identified by the code `S1`…`S12` in every tracked file:
  pipeline source, tests and figure scripts alike.
- `scripts/subject_ids.py` is the only module that defines the format. Parse
  and sort subject codes through it rather than by hand: codes are unpadded,
  so `sorted()` places `S10` before `S2`, and `subject_sort_key` exists to
  fix that.
- Acquisition identifiers belong only in the gitignored
  `config/subject_map.csv`, alongside `.env`. They must never appear in any
  tracked file; `scripts/check_public_hygiene.py` fails the build on any
  that do.
- Do not write a per-subject hash of an acquisition identifier into any
  output. A truncated digest over twelve subjects and a known prefix can be
  brute-forced, so it would publish the mapping rather than protect it.

## Repo layout

```
scripts/           pipeline source
  ├── assessments/   clinical scales (BRIEF-A, PHQ-9, GAD-7, craving, TEA)
  ├── behav/         Stroop, WCST scoring (Heaton rules)
  ├── brain/         fNIRS GLM, step1 to step4 (reads SNIRF; see brain/README.md)
  ├── hypo_test/     Unified EF model, ROI and channel scope (see hypo_test/README.md)
  ├── longitudinal/  longitudinal case series (see longitudinal/README.md)
  ├── figures/       manuscript figures, one script per figure (or main/supplementary pair)
  ├── slurm/         variant_chain.sbatch, the stage chain for one run tree
  └── verification/  compare run trees and summarize variants
config/            analysis variants, probe template
tests/             pytest
```

Data, pipeline outputs and rendered figures are not tracked. `.gitignore`
covers `output/` and `figures/` (except `scripts/figures/`).

## Analysis variants

Every run tree carries `analysis_variant.json` (`scripts/analysis_variant.py`);
the stages from `step3a` onward refuse to run without it. Five variants are in
`config/analysis_variants/`: `main` (the primary analysis), `supp_all_channels`,
`supp_low_accuracy`, `supp_count`, `supp_brain_activation`. Each supplement
changes one decision relative to `main`; the README's "Which code produces
what" section lists them. The figure scripts read the `main` tree only.

## Analysis pipeline

```
1. assessments/00-05                      → Assessment scores (BRIEF-A, PHQ-9, GAD-7, craving, TEA)
2. behav/01_stroop_behav.py               → Stroop behavioral metrics
   behav/01_wcst_behav.py                 → WCST behavioral metrics (Heaton scoring)
3. brain/step1_glm_{stroop,wcst}.py       → First-level GLM (via SLURM array jobs)
4. brain/step2a_glm_to_parcellation.py    → Per-session channel→parcellation mapping
   brain/step2b_parcellation_to_roi.py    → Per-session parcellation→ROI mapping
   brain/step2c_session_info.py           → Per-session short channels, WCST condition counts, low-accuracy sessions
5. brain/step3a_channel_group_analysis.py → Channel-level group statistics
   brain/step3b_roi_extraction.py         → ROI extraction per subject
   brain/step3c_roi_group_analysis.py     → ROI-level group statistics
6. brain/step4_glm_topoplots.py           → Brain visualizations
7. hypo_test/unified_ef_model_roi.py      → ROI-level hypothesis testing (confirmatory)
   hypo_test/unified_ef_model_channel.py  → Channel-level hypothesis testing (exploratory)
8. longitudinal/extract_roi_alltimepoints.py, unified_ef_longitudinal.py,
   longitudinal_exploratory.py            → Longitudinal case series
9. figures/*.py                           → Manuscript figure panels (main tree only)
```

## GLM contrasts

**Stroop**: `stroop_effect` (incongruent - congruent_mixed), `mixing_cost` (congruent_mixed - congruent_pure), `incongruent_vs_pure`
**WCST**: `correct_vs_incorrect`, `change_vs_regular`, `learning`, `duration_effect_correct`

## Hypothesis framework (Unified EF model)

Trait EF (BRIEF-A) → state (fNIRS brain activation) → outcome (task behavior). This is a convergent-validity framework, not formal mediation.

| H | Construct | BRIEF-A | Brain ROI/Contrast | Behavior |
|---|-----------|---------|-------------------|----------|
| H1 | Inhibition | Inhibit | SalVentAttn/Cont (Stroop effect) | Interference |
| H2 | Cognitive Flexibility | Shift | Cont (WCST change vs regular) | Perseverative error rate (count in `supp_count`) |
| H3 | Rule Learning | Working Memory | Cont (WCST learning) | WM maintenance decline |
| H4 | Attentional Stability | Working Memory | Cont (Stroop mixing cost) | RT variability |
| H5 | Global EF | GEC | — | Global performance z-score (Stroop accuracy and proportion of WCST rule periods mastered) |

## Statistical methods

- **Correlations**: Spearman
- **Permutation testing**: p = (1 + count) / (1 + n_perm). Confirmatory ROI tests 10,000 permutations; exploratory channel tests 50,000; GLM channel sign-flip 10,000
- **Bootstrap**: 95% intervals. ROI tests 5,000 resamples, channel tests 1,000 (percentile); GLM Cohen's d 5,000 (BCa)
- **Multiple comparisons**: FDR (Benjamini-Hochberg). The 8 confirmatory ROI tests (self-report × brain and brain × behavior for H1–H4) form one family; exploratory channel tests are corrected within each hypothesis, pooling both sub-pathways; GLM channels are corrected within each contrast. Never re-run FDR on a subset of a family.
- **Effect sizes**: Cohen's d with bootstrap CIs

## Common pitfalls

- **Per-session ROI mapping**: each session records its own set of HbO channels, so the channel-to-ROI mapping is per session, not universal.
- **Baseline-only filter**: all cross-sectional analyses use `task_order=1`. Later timepoints are for the longitudinal case series only.
- **Short channels**: channels under 10 mm are nuisance regressors in the first-level GLM and are dropped from every later stage in `main`.
- **Canonical channel names**: channels are named by optode position in the full probe's label order (`scripts/brain/channel_names.py`); never pool channels across sessions by raw MNE names.
- **Per-contrast vlim for β topomaps**: β scales differ by about a factor of ten across contrasts. Use the 95th percentile of |β| per contrast; a per-task limit washes out the small-β contrasts.

## Technical notes

- **SLURM**: the first-level GLM runs via `stroop_glm_1st.sh` / `wcst_glm_1st.sh` (array jobs); later stages via `scripts/slurm/variant_chain.sbatch`.
- **Data format**: SNIRF files from the Kernel Flow system.
- **6 of 7 networks accessible**: Limbic is excluded (fNIRS depth limitation).

## Tests and validation

- New code: include a pytest test under `tests/`. A test that needs the study data takes a fixture from `tests/conftest.py`, so it skips when the data are absent.
- Run `python scripts/check_public_hygiene.py` before committing; it fails on acquisition identifiers, absolute paths, email addresses and session IDs in tracked files.

## Commit guidelines

- Imperative subject ("add", "fix", "update"), under 70 characters
- Body explains *why*; reference file paths in the body
- Don't commit secrets (`.env`, tokens, anything that could identify a participant)
- Don't commit data, pipeline outputs or rendered figures
