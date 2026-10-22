"""Shared input checks for the figure scripts.

Every published figure is drawn from a run tree built with the ``main``
analysis variant. ``figure_inputs`` refuses any other tree, and any tree
that lacks one of the files a script reads, before a script starts
drawing, so a wrong ``OUTPUT_DIR`` ends with a message rather than a
traceback or a partial set of panels.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_variant import (  # noqa: E402
    VariantError, load_variant, refuse_protected_tree,
)

FIGURE_VARIANT = "main"


def figure_inputs(output_dir: Path, required: list[str]) -> Path:
    """Check a run tree and return the directory figures are written to.

    ``required`` lists paths relative to ``output_dir``. Raises SystemExit
    naming the variant when the tree is not ``main``, and SystemExit listing
    every required path that does not exist. Raises SystemExit when
    ``output_dir`` is a protected reference tree (the PROTECTED_TREE_MARKERS
    setting; see analysis_variant.py), since the scripts write and delete panel files
    under it. Returns ``output_dir / "figures"``.
    """
    output_dir = Path(output_dir)
    try:
        variant = load_variant(output_dir)
    except VariantError as exc:
        raise SystemExit(f"cannot read the analysis variant: {exc}") from None
    if variant.name != FIGURE_VARIANT:
        raise SystemExit(
            f"{output_dir} is the {variant.name!r} variant; figures are drawn "
            f"from a {FIGURE_VARIANT!r} run tree only"
        )
    missing = [rel for rel in required if not (output_dir / rel).exists()]
    if missing:
        listing = "\n".join(f"  {rel}" for rel in missing)
        raise SystemExit(f"{output_dir} lacks {len(missing)} input(s) this "
                         f"figure needs:\n{listing}")
    try:
        refuse_protected_tree(str(output_dir))
    except RuntimeError as exc:
        raise SystemExit(f"{exc}; point OUTPUT_DIR at a copy or a symlinked "
                         "tree with its own figures/ directory") from None
    return output_dir / "figures"


def env_dir(name: str) -> Path:
    """Directory named by environment variable ``name``; SystemExit if unset."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} must be set (environment or .env)")
    return Path(value)
