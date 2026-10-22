"""Which analysis decisions a run tree was built with.

Every run tree carries analysis_variant.json at its root. Stages 05-10 read
it through load_variant(); a missing file, a missing key, an unknown key or
an unknown value is an error, so a tree can never be built with a decision
silently left at its old default.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

VARIANT_FILENAME = "analysis_variant.json"

# Path fragments that mark a run tree as a finished reference come from the
# PROTECTED_TREE_MARKERS environment variable (or the repository's .env): a
# comma-separated list of substrings, e.g. "runs/archive,frozen_outputs".
# Every stage that writes into OUTPUT_DIR (the assessment, behavior, brain,
# hypothesis-test and longitudinal entry points, the figure scripts and
# slurm/variant_chain.sbatch) refuses to run when OUTPUT_DIR contains any of
# them, so a mistyped OUTPUT_DIR cannot overwrite an archived
# result. A substring match also covers every subtree below a marked
# directory. Unset or empty, nothing is protected. Shell callers use
# ``python scripts/analysis_variant.py --check-output-dir "$OUTPUT_DIR"``.
PROTECTED_TREE_MARKERS_VAR = "PROTECTED_TREE_MARKERS"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def protected_tree_markers() -> tuple[str, ...]:
    """The configured markers, read from the environment or the repository's .env.

    A variable already set in the environment wins over .env.
    """
    load_dotenv(_REPO_ROOT / ".env", override=False)
    raw = os.environ.get(PROTECTED_TREE_MARKERS_VAR, "")
    return tuple(m.strip() for m in raw.split(",") if m.strip())

_ALLOWED = {
    "channels": {"all", "long"},
    "h2_metric": {"perseverative_errors", "perseverative_error_rate"},
    "second_timepoint": {"most_complete"},
    "brain_direction": {"efficiency", "activation"},
}
_BOOLS = {"exclude_low_accuracy", "exclude_invalid_brief",
          "mask_wcst_without_regular_incorrect"}


class VariantError(Exception):
    pass


@dataclass(frozen=True)
class AnalysisVariant:
    name: str
    channels: str
    exclude_low_accuracy: bool
    h2_metric: str
    exclude_invalid_brief: bool
    second_timepoint: str
    mask_wcst_without_regular_incorrect: bool
    brain_direction: str


def load_variant(output_dir=None, *, path=None) -> AnalysisVariant:
    if path is None:
        if output_dir is None:
            output_dir = os.environ.get("OUTPUT_DIR")
            if not output_dir:
                raise VariantError("OUTPUT_DIR is not set")
        path = Path(output_dir) / VARIANT_FILENAME
    path = Path(path)
    if not path.is_file():
        raise VariantError(f"{VARIANT_FILENAME} not found at {path}")
    payload = json.loads(path.read_text())
    expected = {f.name for f in fields(AnalysisVariant)}
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise VariantError(f"{path}: unknown key(s) {unknown}")
    if missing:
        raise VariantError(f"{path}: missing key(s) {missing}")
    for key, allowed in _ALLOWED.items():
        if payload[key] not in allowed:
            raise VariantError(f"{path}: {key}={payload[key]!r}, expected one of {sorted(allowed)}")
    for key in _BOOLS:
        if not isinstance(payload[key], bool):
            raise VariantError(f"{path}: {key} must be true or false, got {payload[key]!r}")
    return AnalysisVariant(**payload)


def refuse_protected_tree(output_dir: str) -> None:
    """Raise RuntimeError if ``output_dir`` contains a configured protected-tree marker."""
    for marker in protected_tree_markers():
        if marker in str(output_dir):
            raise RuntimeError(
                f"refusing to write into {output_dir!r}: path contains {marker!r}"
            )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Guard against writing into a protected run tree.")
    parser.add_argument("--check-output-dir", required=True, metavar="DIR")
    try:
        refuse_protected_tree(parser.parse_args().check_output_dir)
    except RuntimeError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(1)
