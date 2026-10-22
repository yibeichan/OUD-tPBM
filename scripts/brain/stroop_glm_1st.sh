#!/bin/bash
# First-level Stroop GLM as a SLURM array job.
# Runs once per row in ${BASE_DIR}/data/stroop_task_info.csv.
#
# Requirements: a micromamba environment named `fnirs` (activated below) and
# BASE_DIR in .env or the environment. No partition is set here: export
# SBATCH_PARTITION=<partition> before running this script to choose one
# (sbatch reads that variable); otherwise the cluster default applies.
# Resource requests (--mem, --time) were sized for the study data.
#
# Run it with bash from the repository root (bash scripts/brain/stroop_glm_1st.sh):
# it submits the array job itself, with one task per row.
#
# OUTPUT_DIR: export it before running (OUTPUT_DIR=/path/to/tree bash ...)
# to choose the run tree; an exported value takes precedence over .env. With
# none exported, .env supplies it. The step1 script refuses protected trees.

# Resolve the repo root. sbatch copies this script into a spool directory
# (e.g. /var/spool/slurmd/job12345/slurm_script) before executing the array
# task, so ${BASH_SOURCE[0]} there points into the spool, not the checkout.
# Prefer an explicit override, then the submission directory (which sbatch
# sets automatically as SLURM_SUBMIT_DIR) if it is a checkout of this repo,
# then fall back to BASH_SOURCE for a direct or --test invocation. A submit
# directory that is not this checkout (e.g. sbatch called from inside a SLURM
# job whose working directory is elsewhere) is skipped.
if [ -n "${PROJECT_REPO:-}" ]; then
    REPO_ROOT="${PROJECT_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && \
     [ -f "${SLURM_SUBMIT_DIR}/scripts/brain/step1_glm_stroop.py" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# Verify the resolved root really is this checkout before using it -- never
# fall through to a wrong path.
if [ ! -f "${REPO_ROOT}/scripts/brain/step1_glm_stroop.py" ]; then
    echo "Error: '${REPO_ROOT}' does not look like the OUD-tPBM checkout" \
         "(scripts/brain/step1_glm_stroop.py not found there)." \
         "Set PROJECT_REPO to the repo root to fix this." >&2
    exit 1
fi

# First log line: the code revision and whether the tree is clean.
echo "commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "tree status (empty = clean): $(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | head -20)"

# Source .env (KEY=value format) so BASE_DIR is available inside the array task.
# An OUTPUT_DIR already exported by the caller wins over the one in .env, so a
# run tree chosen at submission time (OUTPUT_DIR=... bash this script) is the
# one the array tasks write to.
_caller_has_output_dir="${OUTPUT_DIR:+set}"
_caller_output_dir="${OUTPUT_DIR-}"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi
if [ -n "${_caller_has_output_dir}" ]; then
    export OUTPUT_DIR="${_caller_output_dir}"
fi
: "${BASE_DIR:?BASE_DIR must be set (in .env or environment)}"

TASK_NAME="stroop"
FILE_INFO="${BASE_DIR}/data/${TASK_NAME}_task_info.csv"

# Optional notification email — set SLURM_MAIL_USER in your environment to enable.
SBATCH_MAIL_FLAGS=()
if [ -n "${SLURM_MAIL_USER:-}" ]; then
    SBATCH_MAIL_FLAGS=(--mail-type=FAIL,END --mail-user="${SLURM_MAIL_USER}")
fi

# Check if being run directly (not through sbatch)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    if [ "$1" = "--test" ]; then
        echo "Running test mode (first row only)"
        export SLURM_ARRAY_TASK_ID=0
    else
        if [ ! -f "$FILE_INFO" ]; then
            echo "Error: CSV file not found at $FILE_INFO"
            exit 1
        fi

        N_TASKS=$(( $(wc -l < "$FILE_INFO") - 1 ))
        if [ $N_TASKS -lt 1 ]; then
            echo "Error: CSV file appears to be empty or contains only header"
            exit 1
        fi

        echo "Submitting job array with $N_TASKS tasks"

        mkdir -p "${REPO_ROOT}/logs"

        sbatch --job-name=stroop_glm_1st \
               --output="${REPO_ROOT}/logs/stroop_glm_1st-%j.out" \
               --error="${REPO_ROOT}/logs/stroop_glm_1st-%j.err" \
               --array=0-$((N_TASKS-1)) \
               --nodes=1 \
               --ntasks=1 \
               --cpus-per-task=1 \
               --mem=4G \
               --time=00:45:00 \
               --export=ALL,PROJECT_REPO="${REPO_ROOT}" \
               "${SBATCH_MAIL_FLAGS[@]}" \
               "$0"
        exit 0
    fi
fi

# Inside a SLURM job: stagger startup to avoid conda lock contention.
sleep $((SLURM_ARRAY_TASK_ID % 10))

eval "$(micromamba shell hook --shell bash)"
micromamba activate fnirs

python "${REPO_ROOT}/scripts/brain/step1_glm_stroop.py" \
    --task_name "$TASK_NAME" \
    --file_info "$FILE_INFO" \
    --array_idx "${SLURM_ARRAY_TASK_ID}"
