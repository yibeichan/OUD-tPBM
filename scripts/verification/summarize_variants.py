#!/usr/bin/env python3
"""Summarize headline numbers across analysis-variant run trees.

Reads one or more run trees (a `main` tree plus its one-factor supplements,
each carrying `analysis_variant.json` at its root -- see
`scripts/analysis_variant.py`) and writes a single markdown report comparing
their headline results side by side: channel- and ROI-level FDR survivors,
the confirmatory ROI hypothesis-test family, the exploratory channel-level
family, the longitudinal case-series classification, and the excluded
sessions / invalid BRIEF-A administrations feeding into all of the above.

Every number is read from the trees' own CSV/JSON outputs -- nothing here
recomputes a statistic; it only counts, filters and reshapes what the
pipeline already wrote. One tree among ``--trees`` must carry
``analysis_variant.json``'s ``name`` field equal to ``"main"``; every other
tree is labelled by the one (or more) analysis decision where its variant
differs from main's.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from analysis_variant import AnalysisVariant, load_variant  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402

TASKS = ("stroop", "wcst")
BRIEF_ADMINISTRATIONS = ("baseline", "post", "followup")


# ---------------------------------------------------------------------------
# Tree identity and labelling
# ---------------------------------------------------------------------------

def load_tree_variant(tree: Path) -> AnalysisVariant:
    return load_variant(path=tree / "analysis_variant.json")


def find_main(variants: dict[str, AnalysisVariant]) -> AnalysisVariant:
    for variant in variants.values():
        if variant.name == "main":
            return variant
    raise SystemExit(
        "no --trees entry has analysis_variant.json name == 'main'; "
        "a main tree is required to label what each supplement changes"
    )


def decision_diff(variant: AnalysisVariant, main: AnalysisVariant) -> list[str]:
    """Decision fields where ``variant`` differs from ``main`` (name excluded)."""
    diffs = []
    for f in fields(variant):
        if f.name == "name":
            continue
        va, vb = getattr(variant, f.name), getattr(main, f.name)
        if va != vb:
            diffs.append(f"{f.name}={va!r}")
    return diffs


def tree_header(variant: AnalysisVariant, main: AnalysisVariant) -> str:
    if variant.name == main.name:
        return variant.name
    diffs = decision_diff(variant, main)
    if not diffs:
        return variant.name
    return f"{variant.name} ({', '.join(diffs)})"


# ---------------------------------------------------------------------------
# Per-tree readers -- each returns plain pandas objects, no formatting
# ---------------------------------------------------------------------------

def _survivor_counts(tree: Path, relpath_for_task) -> pd.DataFrame:
    """One row per (task, contrast): rows tested and FDR survivors.

    ``relpath_for_task(task)`` returns each task's statistics CSV path
    relative to ``tree``; every such CSV carries ``contrast`` and
    ``significant_fdr`` columns (one row per channel or per ROI).
    """
    rows = []
    for task in TASKS:
        df = pd.read_csv(tree / relpath_for_task(task))
        for contrast, group in df.groupby("contrast"):
            rows.append({
                "task": task,
                "contrast": contrast,
                "n_tested": len(group),
                "n_fdr": int(group["significant_fdr"].sum()),
            })
    return pd.DataFrame(rows)


def channel_survivor_counts(tree: Path) -> pd.DataFrame:
    """Channels tested and FDR survivors per (task, contrast) (05)."""
    return _survivor_counts(
        tree,
        lambda task: Path("05_channel_group_analysis") / task
        / f"{task}_all_contrasts_channel_statistics.csv",
    )


def roi_survivor_counts(tree: Path) -> pd.DataFrame:
    """ROIs tested and FDR survivors per (task, contrast) (07)."""
    return _survivor_counts(
        tree,
        lambda task: Path("07_roi_analysis") / task / "group_roi_statistics_network_roi.csv",
    )


def confirmatory_roi_tests(tree: Path) -> pd.DataFrame:
    """The confirmatory 8-test family: rows with a p_fdr (09, ROI level)."""
    path = tree / "09_hypothesis_testing" / "unified_ef_model_roi" / "results_summary.csv"
    df = pd.read_csv(path)
    return df[df["p_fdr"].notna()].reset_index(drop=True)


def exploratory_channel_family(tree: Path) -> tuple[int, pd.DataFrame]:
    """(family size, survivor rows) for the exploratory channel family (09)."""
    path = tree / "09_hypothesis_testing" / "unified_ef_model_channel" / "results_summary.csv"
    df = pd.read_csv(path)
    survivors = df[df["Significant"] == "Yes"].reset_index(drop=True)
    return len(df), survivors


def second_timepoint_table(tree: Path) -> pd.DataFrame:
    path = tree / "10_longitudinal" / "unified_ef" / "second_timepoint_selection.csv"
    return pd.read_csv(path)


def convergence_table(tree: Path) -> pd.DataFrame:
    path = tree / "10_longitudinal" / "unified_ef" / "hypothesis_convergence.csv"
    return pd.read_csv(path)


def convergence_grid(convergence: pd.DataFrame) -> pd.DataFrame:
    """Per-participant x pathway classification grid (rows=subject, cols=hypothesis)."""
    return convergence.pivot(index="subject_id", columns="hypothesis", values="convergence_class")


def longitudinal_headline(convergence: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(subjects convergent on >=1 pathway, subjects convergent at all 3 levels)."""
    improved = convergence[convergence["convergence_class"] == "convergent_improvement"]
    any_pathway = sorted(improved["subject_id"].unique())
    all_three = sorted(improved.loc[improved["n_domains"] == 3, "subject_id"].unique())
    return any_pathway, all_three


def excluded_sessions(tree: Path) -> pd.DataFrame:
    path = tree / "04b_session_info" / "low_accuracy_sessions.csv"
    return pd.read_csv(path)


def invalid_brief_administrations(tree: Path) -> pd.DataFrame:
    """(administration, subject) rows failing a BRIEF-A validity scale.

    Restricted to administrations with n_items > 0 (an empty administration
    carries no scale scores to fail), per scripts/brief_validity.py.
    """
    rows = []
    for admin in BRIEF_ADMINISTRATIONS:
        path = (
            tree / "01_assessments" / "t-scores"
            / f"executive_function_{admin}_subscale_t-scores.csv"
        )
        df = pd.read_csv(path)
        df = df[df["n_items"] > 0]
        if df.empty:
            continue
        mask = valid_administration_mask(df)
        for subject in df.loc[~mask, "subject"]:
            rows.append({"administration": admin, "subject": subject})
    return pd.DataFrame(rows, columns=["administration", "subject"])


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "NaN"
        return f"{value:.4g}"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _table(columns: list[str], rows: list[list]) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _df_table(df: pd.DataFrame) -> str:
    return _table(list(df.columns), df.to_numpy().tolist())


def _gated_exclusion_section(
    heading: str, *, flag_name: str, reader, empty_note: str,
    tree_paths: dict[str, Path], variants: dict[str, AnalysisVariant],
    headers: dict[str, str], keys: list[str],
) -> list[str]:
    """Render one tree-by-tree exclusion table, gated on the tree's own flag.

    ``reader(tree_path)`` returns every row the underlying rule flags,
    independent of whether the run's variant actually excludes them.
    When the tree's ``flag_name`` is false, the rule was not applied there:
    the section says so and, if there are rows, labels them "not excluded"
    rather than implying they were dropped from that tree's analyses.
    """
    lines = [heading, ""]
    for key in keys:
        applied = getattr(variants[key], flag_name)
        lines.append(f"**{headers[key]}**")
        lines.append("")
        if not applied:
            lines.append(f"Rule not applied in this tree ({flag_name}=false).")
            lines.append("")
        df = reader(tree_paths[key]).copy()
        if df.empty:
            lines.append(empty_note)
            lines.append("")
        else:
            df.insert(0, "status", "excluded" if applied else "not excluded")
            lines.append(_df_table(df))
    return lines


def _union_index(counts: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Row keys across every tree's counts, in first-seen order.

    Trees are expected to test the same (task, contrast) set, but this
    does not assume the first tree in ``keys`` has the full set -- a
    supplement missing a row (or main missing one a supplement has) still
    gets a row in the table, filled with ``--`` for the side missing it.
    """
    combined = pd.concat([df[columns] for df in counts.values()], ignore_index=True)
    return combined.drop_duplicates().reset_index(drop=True)


def _union_first_seen(name_lists) -> list[str]:
    seen: list[str] = []
    known = set()
    for names in name_lists:
        for name in names:
            if name not in known:
                known.add(name)
                seen.append(name)
    return seen


def _render_survivor_table(
    heading: str, counts: dict[str, pd.DataFrame], keys: list[str], headers: dict[str, str]
) -> list[str]:
    """Render one "tested / FDR survivors" section shared by 05 and 07.

    ``counts[key]`` is a per-(task, contrast) table as returned by
    ``channel_survivor_counts`` / ``roi_survivor_counts``; row keys are the
    union across all trees (see ``_union_index``), so a (task, contrast)
    present in only some trees still gets a row, ``--`` where absent.
    """
    lines = [heading, ""]
    index = _union_index(counts, ["task", "contrast"])
    rows = []
    for _, idx_row in index.iterrows():
        row = [idx_row["task"], idx_row["contrast"]]
        for key in keys:
            df = counts[key]
            match = df[(df["task"] == idx_row["task"]) & (df["contrast"] == idx_row["contrast"])]
            if match.empty:
                row.append("--")
            else:
                m = match.iloc[0]
                row.append(f"{m['n_fdr']}/{m['n_tested']}")
        rows.append(row)
    lines.append(_table(["task", "contrast"] + [headers[k] for k in keys], rows))
    return lines


def _test_cell(row: pd.Series) -> str:
    return (
        f"r={_fmt(row['r'])} (CI {_fmt(row['CI_lower'])} to {_fmt(row['CI_upper'])}), "
        f"p={_fmt(row['p'])}, p_fdr={_fmt(row['p_fdr'])}, n={_fmt(row['n'])}"
    )


def build_report(tree_paths: dict[str, Path]) -> str:
    """``tree_paths`` maps a display key (e.g. CLI arg order) to a tree root.

    The map's iteration order sets column order in every table. Labels
    shown to the reader come from each tree's own analysis_variant.json,
    not from ``tree_paths``' keys.
    """
    variants = {key: load_tree_variant(path) for key, path in tree_paths.items()}
    main_variant = find_main(variants)
    headers = {key: tree_header(v, main_variant) for key, v in variants.items()}
    keys = list(tree_paths)

    lines: list[str] = []
    lines.append("# Analysis variant comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Trees compared")
    lines.append("")
    for key in keys:
        lines.append(f"- {headers[key]}: `{tree_paths[key]}`")
    lines.append("")

    # --- Analysis decisions -----------------------------------------------
    lines.append("## Analysis decisions")
    lines.append("")
    decision_fields = [f.name for f in fields(main_variant) if f.name != "name"]
    rows = [
        [field] + [_fmt(getattr(variants[key], field)) for key in keys]
        for field in decision_fields
    ]
    lines.append(_table(["decision"] + [headers[k] for k in keys], rows))

    # --- Channel-level group statistics (05) -------------------------------
    channel_counts = {key: channel_survivor_counts(tree_paths[key]) for key in keys}
    lines.extend(_render_survivor_table(
        "## Channel-level group statistics (05) -- tested / FDR survivors",
        channel_counts, keys, headers,
    ))

    # --- ROI-level group statistics (07) -----------------------------------
    roi_counts = {key: roi_survivor_counts(tree_paths[key]) for key in keys}
    lines.extend(_render_survivor_table(
        "## ROI-level group statistics (07) -- tested / FDR survivors",
        roi_counts, keys, headers,
    ))

    # --- Confirmatory ROI hypothesis tests (09, roi) -----------------------
    lines.append(
        "## Confirmatory ROI hypothesis tests "
        "(09_hypothesis_testing/unified_ef_model_roi) -- 8-test family"
    )
    lines.append("")
    confirmatory = {key: confirmatory_roi_tests(tree_paths[key]) for key in keys}
    test_names = _union_first_seen(confirmatory[key]["Test"] for key in keys)
    rows = []
    for test_name in test_names:
        row = [test_name]
        for key in keys:
            df = confirmatory[key]
            match = df[df["Test"] == test_name]
            row.append(_test_cell(match.iloc[0]) if not match.empty else "--")
        rows.append(row)
    lines.append(_table(["Test"] + [headers[k] for k in keys], rows))

    # --- Exploratory channel-level family (09, channel) --------------------
    lines.append(
        "## Exploratory channel-level hypothesis tests "
        "(09_hypothesis_testing/unified_ef_model_channel)"
    )
    lines.append("")
    exploratory = {key: exploratory_channel_family(tree_paths[key]) for key in keys}
    rows = [["family size / FDR survivors"] + [
        f"{len(exploratory[key][1])}/{exploratory[key][0]}" for key in keys
    ]]
    lines.append(_table(["metric"] + [headers[k] for k in keys], rows))
    lines.append("")
    lines.append("Survivor tests:")
    lines.append("")
    for key in keys:
        _, survivors = exploratory[key]
        lines.append(f"- {headers[key]}:")
        if survivors.empty:
            lines.append("  - none")
        else:
            for _, srow in survivors.iterrows():
                lines.append(f"  - {srow['Test']}: {_test_cell(srow)}")
    lines.append("")

    # --- Longitudinal --------------------------------------------------
    lines.append("## Longitudinal (10_longitudinal/unified_ef)")
    lines.append("")
    lines.append("### Second-timepoint selection")
    lines.append("")
    t2_tables = {key: second_timepoint_table(tree_paths[key]) for key in keys}
    for key in keys:
        lines.append(f"**{headers[key]}**")
        lines.append("")
        lines.append(_df_table(t2_tables[key]))

    lines.append("### Per-participant x pathway convergence")
    lines.append("")
    convergence_tables = {key: convergence_table(tree_paths[key]) for key in keys}
    for key in keys:
        lines.append(f"**{headers[key]}**")
        lines.append("")
        grid = convergence_grid(convergence_tables[key]).reset_index()
        lines.append(_df_table(grid))

    lines.append("### Headline counts")
    lines.append("")
    headline = {key: longitudinal_headline(convergence_tables[key]) for key in keys}
    rows = [
        ["participants convergent on >=1 pathway"] + [
            f"{len(headline[key][0])} ({', '.join(headline[key][0]) or '--'})" for key in keys
        ],
        ["participants convergent at all three levels"] + [
            f"{len(headline[key][1])} ({', '.join(headline[key][1]) or '--'})" for key in keys
        ],
    ]
    lines.append(_table(["metric"] + [headers[k] for k in keys], rows))

    # --- Excluded sessions and invalid BRIEF-A -----------------------------
    lines.extend(_gated_exclusion_section(
        "## Excluded sessions (04b_session_info/low_accuracy_sessions.csv)",
        flag_name="exclude_low_accuracy",
        reader=excluded_sessions,
        empty_note="_(no sessions at or below chance accuracy)_",
        tree_paths=tree_paths, variants=variants, headers=headers, keys=keys,
    ))

    lines.extend(_gated_exclusion_section(
        "## Invalid BRIEF-A administrations",
        flag_name="exclude_invalid_brief",
        reader=invalid_brief_administrations,
        empty_note="_(no invalid administrations)_",
        tree_paths=tree_paths, variants=variants, headers=headers, keys=keys,
    ))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True, help="Run tree roots to compare.")
    parser.add_argument("--out", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    resolved = [Path(t).resolve() for t in args.trees]
    if len(set(resolved)) != len(resolved):
        raise SystemExit(f"--trees lists the same path more than once: {args.trees}")

    # Keyed by position: a tree's display label comes from its own
    # analysis_variant.json (read inside build_report), not from this key.
    tree_paths = {str(i): Path(t) for i, t in enumerate(args.trees)}
    report = build_report(tree_paths)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
