#!/bin/bash
# Run a chain of pipeline stages (2c ... long_expl) on one run tree.
#
# Submit from the repository root:
#   RUN_TREE=/path/to/run_tree STAGES="2c 3a 3b 3c 4" sbatch scripts/slurm/variant_chain.sbatch
#
# Required environment:
#   RUN_TREE   run tree to write into (must already hold analysis_variant.json)
#   STAGES     space-separated stage list, in order:
#              2c 3a 3b 3c 4 roi10 hypo_roi hypo_channel long long_expl
# Logs go to logs/ under the repository root (tracked via logs/.gitkeep, so
# it exists in a fresh clone; Slurm opens the log files before this script
# runs). To put them elsewhere, pass sbatch --output=... --error=... instead.
# The repository's .env supplies BASE_DIR. Needs the `fnirs` micromamba env.
# No partition is set here: export SBATCH_PARTITION=<partition> before
# submitting to choose one (sbatch reads that variable); otherwise the
# cluster default applies. The memory and time lines were sized for the
# study data.
#SBATCH --job-name=oudtpbm_variant
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/variant-%x-%j.out
#SBATCH --error=logs/variant-%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f scripts/slurm/variant_chain.sbatch ] || { echo "ABORT: submit from the repo root"; exit 1; }
echo "commit: $(git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD)"
echo "tree status (empty = clean): $(git -C "$SLURM_SUBMIT_DIR" status --porcelain | head -20)"
eval "$(micromamba shell hook -s bash)"; micromamba activate fnirs
set -a; . ./.env; set +a
: "${RUN_TREE:?set RUN_TREE}"; : "${STAGES:?set STAGES}"
export OUTPUT_DIR="$RUN_TREE"            # after .env, which would otherwise win
python scripts/analysis_variant.py --check-output-dir "$OUTPUT_DIR"   # protected-tree guard
[ -f "$OUTPUT_DIR/analysis_variant.json" ] || { echo "ABORT: no analysis_variant.json"; exit 1; }
echo "variant:"; cat "$OUTPUT_DIR/analysis_variant.json"
S=20260922
run () { echo; echo "===== $* ====="; date -Is; python "$@"; echo "ok: $* at $(date -Is)"; }
for st in $STAGES; do case $st in
  2c) run scripts/brain/step2c_session_info.py ;;
  3a) for t in wcst stroop; do run scripts/brain/step3a_channel_group_analysis.py --task $t --seed $S; done ;;
  3b) for t in wcst stroop; do run scripts/brain/step3b_roi_extraction.py --task $t --baseline-only; done ;;
  3c) for t in wcst stroop; do run scripts/brain/step3c_roi_group_analysis.py --task $t --seed $S; done ;;
  4)  for t in wcst stroop; do run scripts/brain/step4_glm_topoplots.py --task $t; done ;;
  roi10) run scripts/longitudinal/extract_roi_alltimepoints.py --task both ;;
  hypo_roi) run scripts/hypo_test/unified_ef_model_roi.py --seed $S ;;
  hypo_channel) run scripts/hypo_test/unified_ef_model_channel.py --seed $S ;;
  long) for t in stroop wcst; do p="$OUTPUT_DIR/10_longitudinal/roi_activations_alltimepoints/$t/all_subjects_roi_activation_network_roi.csv"; [ -s "$p" ] || { echo "ABORT: missing $p"; exit 1; }; done
        run scripts/longitudinal/unified_ef_longitudinal.py --seed $S ;;
  long_expl) run scripts/longitudinal/longitudinal_exploratory.py ;;
  *) echo "unknown stage $st"; exit 1;;
esac; done
echo "CHAIN COMPLETE $(date -Is)"
