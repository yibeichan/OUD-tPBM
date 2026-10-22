#!/usr/bin/env python3
"""Set up a run tree for one analysis variant.

Each analysis variant gets its own tree (``--dest``), a sibling of the main
tree, which must be named ``main`` (``MAIN_TREE_NAME``).
Stages 01-04 (assessments, behavior, first-level GLM, channel mappings)
never change across variants, so a variant's tree symlinks them from a
read-only source tree (``--source``) rather than recomputing them. ``04b_session_info`` (short channels,
WCST condition counts, low-accuracy sessions) is likewise shared -- it is
built once, in the main tree, and every other variant's tree symlinks it
from there rather than rebuilding it.

Later stages (05 onward) do depend on the variant and are computed fresh
for most trees, but two variants (``supp_count``, ``supp_brain_activation``)
only change stage 09/10 relative to main and reuse main's 05-08 (and, for
``supp_count``, 10's ROI-activation extraction) via ``--link-from``.

This script only arranges symlinks, copies the variant config, and writes a
README; it never computes analysis outputs itself.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from analysis_variant import VariantError, load_variant

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "analysis_variants"

# Fixed, variant-independent stage directories shared by every run tree.
SHARED_SOURCE_DIRS = ["01_assessments", "02_behav", "03_glm", "04_channel_mappings"]
SESSION_INFO_DIR = "04b_session_info"
MAIN_TREE_NAME = "main"


class SetupError(Exception):
    pass


def _git(*args: str) -> str:
    """Run git in the repo; return "unknown" when this is not a git checkout."""
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _find_stage_dir(root: Path, prefix: str) -> Path:
    matches = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{prefix}_")
    )
    if not matches:
        raise SetupError(f"no directory under {root} starts with '{prefix}_'")
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise SetupError(f"ambiguous stage '{prefix}' under {root}: {names}")
    return matches[0]


def setup_run_tree(
    variant: str,
    source: Path | str,
    dest: Path | str,
    link_from: Path | str | None = None,
    link_stages: list[str] | None = None,
) -> None:
    """Create ``dest`` as a run tree for ``variant``, sourced from ``source``.

    Raises ``SetupError`` for every precondition failure (non-empty dest,
    missing source directory, missing sibling main tree, missing/ambiguous
    link-stage directory, missing or malformed variant config) before
    touching the filesystem.
    """
    source = Path(source)
    dest = Path(dest)
    link_stages = list(link_stages or [])

    if dest.exists() and any(dest.iterdir()):
        raise SetupError(f"refusing to use non-empty dest: {dest}")

    for name in SHARED_SOURCE_DIRS:
        if not (source / name).is_dir():
            raise SetupError(f"missing source directory: {source / name}")

    variant_config = CONFIG_DIR / f"{variant}.json"
    if not variant_config.is_file():
        raise SetupError(f"no analysis variant config: {variant_config}")
    try:
        load_variant(path=variant_config)
    except VariantError as exc:
        raise SetupError(f"malformed variant config {variant_config}: {exc}") from exc

    is_main = variant == "main"
    main_session_info = None
    if not is_main:
        main_tree = dest.parent / MAIN_TREE_NAME
        if not main_tree.is_dir():
            raise SetupError(
                f"{MAIN_TREE_NAME} not found at {main_tree}; build the main "
                "tree (with 04b_session_info) before any other variant"
            )
        main_session_info = main_tree / SESSION_INFO_DIR
        if not main_session_info.is_dir():
            raise SetupError(
                f"{main_tree} has no {SESSION_INFO_DIR}; run step2c_session_info.py "
                "on the main tree before any other variant"
            )

    if link_stages and link_from is None:
        raise SetupError("--link-stages requires --link-from")
    link_from_path = Path(link_from) if link_from is not None else None

    # Resolve every link-stage source before creating anything.
    resolved_links: list[tuple[str, Path]] = []  # (dest-relative path, source dir)
    for stage in link_stages:
        if stage == "10":
            stage_dir = _find_stage_dir(link_from_path, "10")
            src = stage_dir / "roi_activations_alltimepoints"
            if not src.is_dir():
                raise SetupError(f"missing {src}")
            resolved_links.append(
                (f"{stage_dir.name}/roi_activations_alltimepoints", src)
            )
        else:
            stage_dir = _find_stage_dir(link_from_path, stage)
            resolved_links.append((stage_dir.name, stage_dir))

    # All preconditions checked -- now make the tree.
    dest.mkdir(parents=True, exist_ok=True)

    for name in SHARED_SOURCE_DIRS:
        (dest / name).symlink_to((source / name).resolve(), target_is_directory=True)

    if not is_main:
        (dest / SESSION_INFO_DIR).symlink_to(
            main_session_info.resolve(), target_is_directory=True
        )

    shutil.copyfile(variant_config, dest / "analysis_variant.json")

    for rel, src in resolved_links:
        link_path = dest / rel
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(src.resolve(), target_is_directory=True)

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "HEAD")
    readme = [
        f"# Run tree: {dest.name}",
        "",
        f"variant: {variant}",
        f"source: {source}",
        f"link_from: {link_from_path if link_from_path is not None else 'none'}",
        f"link_stages: {' '.join(link_stages) if link_stages else 'none'}",
        f"branch: {branch}",
        f"commit: {commit}",
        "",
    ]
    (dest / "README.md").write_text("\n".join(readme))


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up a variant's run tree.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--link-from", default=None)
    parser.add_argument("--link-stages", nargs="+", default=[])
    args = parser.parse_args()

    try:
        setup_run_tree(
            args.variant, args.source, args.dest, args.link_from, args.link_stages
        )
    except SetupError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
