#!/usr/bin/env python3
"""Compare two run trees, stage by stage.

Walks the named stage directories (or, with ``--only``, exact relative
paths) under both trees, following symlinks, and reports every CSV and
JSON file as byte-equal, equal after dropping the ignored keys/columns, or
different -- plus any file that exists on only one side. Differences in a
CSV are reported per numeric column (max absolute difference and count of
differing cells); other columns are reported as a differing-cell count.
PNG and other non-CSV/JSON files are listed but not compared.

Exits 1 if any file differs (after ignores) or exists on only one side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Only the keys and columns that name the run or when it ran. Anything that
# carries a result, such as the H2 measure, is compared unless the caller
# passes it explicitly.
DEFAULT_IGNORE_JSON_KEYS = ["generated_at", "timestamp", "variant"]
DEFAULT_IGNORE_CSV_COLUMNS = ["variant"]

_MISSING = object()


def _stage_walk_roots(a: Path, b: Path, stages: list[str]) -> tuple[list[str], list[str]]:
    """Top-level directory names in either tree starting with a stage prefix.

    Returns (names, unmatched) where ``unmatched`` lists the requested
    prefixes that matched no top-level directory under either tree -- a
    typo'd or nonexistent stage, as opposed to a stage present on only one
    side (which the file-level A-only/B-only reporting already catches).
    """
    names: set[str] = set()
    unmatched: list[str] = []
    for s in stages:
        matched = [
            p.name
            for root in (a, b) if root.is_dir()
            for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{s}_")
        ]
        if matched:
            names.update(matched)
        else:
            unmatched.append(s)
    return sorted(names), unmatched


def _unmatched_only(a: Path, b: Path, only: list[str]) -> list[str]:
    """``--only`` relpaths that exist under neither tree."""
    return [rel for rel in only if not (a / rel).exists() and not (b / rel).exists()]


def _under_any(relpath: Path, roots: list[Path]) -> bool:
    parts = relpath.parts
    return any(parts[: len(r.parts)] == r.parts for r in roots)


def _collect_files(tree_root: Path, walk_roots: list[str], exclude: list[str]) -> dict[Path, Path]:
    """{relpath from tree_root -> absolute path} for every file under walk_roots."""
    exclude_paths = [Path(e) for e in exclude]
    files: dict[Path, Path] = {}
    for rel_root in walk_roots:
        start = tree_root / rel_root
        if not start.exists():
            continue
        if start.is_file():
            rel = Path(rel_root)
            if not _under_any(rel, exclude_paths):
                files[rel] = start
            continue
        for dirpath, _dirnames, filenames in os.walk(start, followlinks=True):
            for fname in filenames:
                abspath = Path(dirpath) / fname
                rel = abspath.relative_to(tree_root)
                if _under_any(rel, exclude_paths):
                    continue
                files[rel] = abspath
    return files


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    return df.drop(columns=present) if present else df


def _align_columns_by_name(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Reorder ``df_b``'s columns to ``df_a``'s order when the column sets
    match but the order does not, so a reordering alone is not reported as
    a shape mismatch."""
    if list(df_a.columns) != list(df_b.columns) and set(df_a.columns) == set(df_b.columns):
        return df_b[list(df_a.columns)]
    return df_b


def _numeric_cell_stats(sa: pd.Series, sb: pd.Series) -> tuple[int, float]:
    a = sa.to_numpy(dtype=float)
    b = sb.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    mismatched_nan = np.isnan(a) ^ np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    diff[mismatched_nan] = np.nan
    differing = (diff > 0) | mismatched_nan
    n_diff = int(np.sum(differing))
    finite_diff = diff[differing & np.isfinite(diff)]
    max_diff = float(finite_diff.max()) if finite_diff.size else float("nan") if n_diff else 0.0
    return n_diff, max_diff


def _non_numeric_cell_count(sa: pd.Series, sb: pd.Series) -> int:
    a = sa.fillna("<NA>").astype(str)
    b = sb.fillna("<NA>").astype(str)
    return int((a != b).sum())


def _compare_csv(path_a: Path, path_b: Path, ignore_columns: list[str]) -> str:
    if path_a.read_bytes() == path_b.read_bytes():
        return "byte-equal"
    df_a = _drop_columns(pd.read_csv(path_a), ignore_columns)
    df_b = _drop_columns(pd.read_csv(path_b), ignore_columns)
    df_b = _align_columns_by_name(df_a, df_b)
    if df_a.equals(df_b):
        return "equal-after-ignores"
    if list(df_a.columns) != list(df_b.columns) or len(df_a) != len(df_b):
        return (
            f"different (shape mismatch: A={df_a.shape} B={df_b.shape}, "
            f"columns A={list(df_a.columns)} B={list(df_b.columns)})"
        )
    details = []
    for col in df_a.columns:
        sa, sb = df_a[col], df_b[col]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            n_diff, max_diff = _numeric_cell_stats(sa, sb)
            if n_diff:
                details.append(f"{col}: max|diff|={max_diff:.6g}, {n_diff} cell(s) differ")
        else:
            n_diff = _non_numeric_cell_count(sa, sb)
            if n_diff:
                details.append(f"{col}: {n_diff} cell(s) differ")
    if not details:
        return "equal-after-ignores"
    return "different (" + "; ".join(details) + ")"


def _drop_json_keys(data, keys: list[str]):
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in keys}
    return data


def _compare_json(path_a: Path, path_b: Path, ignore_keys: list[str]) -> str:
    if path_a.read_bytes() == path_b.read_bytes():
        return "byte-equal"
    data_a = _drop_json_keys(json.loads(path_a.read_text()), ignore_keys)
    data_b = _drop_json_keys(json.loads(path_b.read_text()), ignore_keys)
    if data_a == data_b:
        return "equal-after-ignores"
    if isinstance(data_a, dict) and isinstance(data_b, dict):
        keys = sorted(set(data_a) | set(data_b))
        differing = [k for k in keys if data_a.get(k, _MISSING) != data_b.get(k, _MISSING)]
        return f"different (keys: {differing})"
    return "different"


def compare_run_trees(
    a: Path | str,
    b: Path | str,
    stages: list[str] | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    ignore_json_keys: list[str] | None = None,
    ignore_csv_columns: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return (has_difference, report_lines)."""
    a, b = Path(a), Path(b)
    exclude = exclude or []
    ignore_json_keys = DEFAULT_IGNORE_JSON_KEYS if ignore_json_keys is None else ignore_json_keys
    ignore_csv_columns = (
        DEFAULT_IGNORE_CSV_COLUMNS if ignore_csv_columns is None else ignore_csv_columns
    )

    lines: list[str] = []
    has_difference = False

    if only:
        walk_roots = list(only)
        for rel in _unmatched_only(a, b, only):
            lines.append(f"ERROR: --only path matches neither tree: {rel}")
            has_difference = True
    else:
        walk_roots, unmatched_stages = _stage_walk_roots(a, b, stages or [])
        for s in unmatched_stages:
            lines.append(
                f"ERROR: --stages prefix matches no directory under either tree: {s!r}"
            )
            has_difference = True

    files_a = _collect_files(a, walk_roots, exclude)
    files_b = _collect_files(b, walk_roots, exclude)

    all_rel = sorted(set(files_a) | set(files_b), key=lambda p: p.as_posix())
    counts = {
        "byte-equal": 0, "equal-after-ignores": 0, "different": 0,
        "skipped (binary)": 0, "one-sided": 0,
    }

    for rel in all_rel:
        posix = rel.as_posix()
        if rel not in files_a:
            lines.append(f"B-only: {posix}")
            has_difference = True
            counts["one-sided"] += 1
            continue
        if rel not in files_b:
            lines.append(f"A-only: {posix}")
            has_difference = True
            counts["one-sided"] += 1
            continue

        suffix = rel.suffix.lower()
        if suffix == ".csv":
            status = _compare_csv(files_a[rel], files_b[rel], ignore_csv_columns)
        elif suffix == ".json":
            status = _compare_json(files_a[rel], files_b[rel], ignore_json_keys)
        else:
            lines.append(f"skipped (binary): {posix}")
            counts["skipped (binary)"] += 1
            continue

        lines.append(f"{status}: {posix}")
        if status.startswith("different"):
            has_difference = True
            counts["different"] += 1
        elif status in counts:
            counts[status] += 1

    lines.append(
        f"compared {len(all_rel)} file(s): "
        f"{counts['byte-equal']} byte-equal, "
        f"{counts['equal-after-ignores']} equal-after-ignores, "
        f"{counts['different']} different, "
        f"{counts['skipped (binary)']} skipped (binary), "
        f"{counts['one-sided']} one-sided"
    )

    return has_difference, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two run trees.")
    parser.add_argument("a", metavar="A")
    parser.add_argument("b", metavar="B")
    parser.add_argument("--stages", nargs="+", default=[])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--ignore-json-keys", nargs="*", default=DEFAULT_IGNORE_JSON_KEYS,
                        help="JSON keys to drop before comparing (default: %(default)s; "
                             "give the flag with no keys to drop none)")
    parser.add_argument("--ignore-csv-columns", nargs="*", default=DEFAULT_IGNORE_CSV_COLUMNS,
                        help="CSV columns to drop before comparing (default: %(default)s; "
                             "give the flag with no columns to drop none)")
    args = parser.parse_args()

    has_difference, lines = compare_run_trees(
        args.a, args.b,
        stages=args.stages, only=args.only, exclude=args.exclude,
        ignore_json_keys=args.ignore_json_keys, ignore_csv_columns=args.ignore_csv_columns,
    )
    print("\n".join(lines))
    sys.exit(1 if has_difference else 0)


if __name__ == "__main__":
    main()
