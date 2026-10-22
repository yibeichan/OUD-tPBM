#!/usr/bin/env python3
"""
Figure 3 and Supplementary Figure S2: longitudinal behavioural measures for
the participants with a second timepoint.

Panel files for manual assembly (no on-figure task labels, no panel
letters, no figure titles):

OUTPUT_DIR/figures/main/Figure3_behavior/
  fig5_a_slopegraph_stroop.{pdf,png}
      3x3 grid, 9 Stroop mini-panels. Each panel = baseline -> time 2 lines
      per participant in raw units. Direction caret (down = lower better,
      up = higher better) prefixes the sub-label; raw values are NOT flipped.
  fig5_b_slopegraph_wcst.{pdf,png}
      3x3 grid, 9 WCST mini-panels, including the perseverative error rate
      (perseverative errors / experimental trials, shown in %), the H2
      behavioural measure of the main analysis. Same per-panel design.
  fig5_c_subject_legend.{pdf,png}
      Subject ID -> marker shape (grey lines, one shape per participant;
      same shapes as Figure 2).

OUTPUT_DIR/figures/supplementary/FigureS2_behavior_change_heatmaps/
  figS_behavior_change_heatmap_a_stroop.{pdf,png}
      9 rows (Stroop metrics) x one column per participant with a Stroop
      pair. Cell = direction-aligned Delta (post - pre, sign-flipped so
      positive = improvement on every measure), divided by max|Delta| within
      metric so the diverging colormap is comparable across rows with very
      different units.
  figS_behavior_change_heatmap_b_wcst.{pdf,png}
      9 rows (WCST metrics) x 5 participant columns. Same encoding.
  figS_behavior_change_heatmap_c_colorbar.{pdf,png}
      Vertical colour bar, PuOr (purple = improvement, orange = worsening,
      range [-1, +1]); PuOr rather than RdBu so that red/blue mean
      positive/negative activation throughout the manuscript.

Sessions: a baseline session at or below chance accuracy is excluded from
the main analysis, so behavior_long.csv has no rows for it. STROOP_SUBJECTS
lists the participants with a Stroop pair; the Stroop panels drop the
others entirely rather than showing them as missing.

Second timepoint: behavior_long.csv already applies each participant's
selected second timepoint (the later session with the most usable levels,
ties to post) and relabels it task_order=2 upstream, in
scripts/longitudinal/longitudinal_exploratory.py (read_task_info /
apply_second_timepoint), so this script does not need its own followup
override.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D

from _common import env_dir, figure_inputs

load_dotenv()

BEHAV_LONG = "10_longitudinal/behavior_long.csv"
CONSISTENCY = "10_longitudinal/consistency_summary.csv"
# Inputs relative to OUTPUT_DIR.
REQUIRED = [BEHAV_LONG, CONSISTENCY]
# Set in main() from OUTPUT_DIR.
FIG_OUT: Path
FIG_SUPP_OUT: Path
BEHAV_LONG_CSV: Path
CONSISTENCY_CSV: Path

LONGITUDINAL_SUBJECTS = ["S1", "S3", "S5", "S7", "S10"]
# Participants with a Stroop pair. Participants whose baseline Stroop session
# is excluded have no Stroop rows in behavior_long.csv; Stroop panels use this
# list so they are absent, not a hatched "missing" column. WCST panels use the
# full LONGITUDINAL_SUBJECTS.
STROOP_SUBJECTS = ["S1", "S3", "S5", "S10"]

# (task, metric_key, display_label, favourable_direction).
STROOP_METRICS: list[tuple[str, str, str, str]] = [
    ("stroop", "overall_rt",                   "Overall RT (s)",              "lower"),
    ("stroop", "overall_accuracy",             "Overall accuracy",            "higher"),
    ("stroop", "overall_error_rate",           "Overall error rate",          "lower"),
    ("stroop", "overall_rt_cv",                "Overall RT CV",               "lower"),
    ("stroop", "stroop_effect_rt",             "Stroop effect RT (s)",        "lower"),
    ("stroop", "stroop_effect_normalized",     "Stroop effect (norm)",        "lower"),
    ("stroop", "mixing_cost_rt",               "Mixing cost RT (s)",          "lower"),
    ("stroop", "mixing_cost_normalized",       "Mixing cost (norm)",          "lower"),
    ("stroop", "conflict_adaptation_cost_rt",  "Conflict adapt. cost RT (s)", "lower"),
]
WCST_METRICS: list[tuple[str, str, str, str]] = [
    ("wcst", "overall_accuracy",             "Overall accuracy",            "higher"),
    ("wcst", "exp_accuracy",                 "Exp-block accuracy",          "higher"),
    ("wcst", "categories_completed",         "Categories completed",        "higher"),
    ("wcst", "set_shifting_efficiency",      "Set-shifting efficiency",     "higher"),
    ("wcst", "mean_trials_to_learn_rule",    "Mean trials to learn rule",   "lower"),
    ("wcst", "perseverative_error_rate",     "Perseverative error rate (%)", "lower"),
    ("wcst", "perseverative_errors",         "Perseverative errors",        "lower"),
    ("wcst", "non_perseverative_errors",     "Non-perseverative errors",    "lower"),
    ("wcst", "exp_rt_cv",                    "Exp-block RT CV",             "lower"),
]
ALL_METRICS = STROOP_METRICS + WCST_METRICS

# Display scaling (does not change the sign of any change): the rate is
# stored as a proportion and drawn as a percentage.
DISPLAY_SCALE: dict[tuple[str, str], float] = {
    ("wcst", "perseverative_error_rate"): 100.0,
}

# Subject encoding: every line is the same neutral grey and each participant
# has a marker shape of his own, so colour does not encode participant. Same
# constants in fig2_figS1_self_report.py.
SUBJECT_LINE_COLOR = "#9a9a9a"
SUBJECT_MARKER_COLOR = "#3a3a3a"
SUBJECT_MARKER = {
    "S1": "o",   # circle
    "S3": "s",   # square
    "S5": "^",   # triangle up
    "S7": "D",   # diamond
    "S10": "v",  # triangle down
}
assert len(set(SUBJECT_MARKER.values())) == len(LONGITUDINAL_SUBJECTS)


SUBJECT_ORDER = LONGITUDINAL_SUBJECTS


# Markers whose values lie within DODGE_TOL of the y-axis span of each other
# would hide one another (e.g. two scores of 0). They are spread
# horizontally by DODGE_STEP (x is categorical: 0 = baseline, 1 = time 2);
# the y values are unchanged.
DODGE_TOL = 0.04
DODGE_STEP = 0.07


def dodge_offsets(values: dict[str, float], span: float) -> dict[str, float]:
    """{subject: x offset} for one time point; 0 for markers not crowded."""
    order = sorted(values, key=lambda s: values[s])
    clusters: list[list[str]] = []
    for subj in order:
        if clusters and values[subj] - values[clusters[-1][-1]] <= DODGE_TOL * span:
            clusters[-1].append(subj)
        else:
            clusters.append([subj])
    out: dict[str, float] = {}
    for cl in clusters:
        cl = sorted(cl, key=SUBJECT_ORDER.index)
        for k, subj in enumerate(cl):
            out[subj] = (k - (len(cl) - 1) / 2) * DODGE_STEP
    return out


def subject_short(subject_id: str) -> str:
    """Identity: subject IDs are already S-codes (S1, S3, S5, S7, S10)."""
    return subject_id

# Diverging map for the change heatmaps: PuOr, purple =
# improvement, orange = worsening. Red/blue are kept for positive/negative
# activation in the brain maps (Figures 1 and S3-S4), so they mean one thing
# throughout the manuscript.
CMAP_CHANGE = "PuOr"
CHANGE_LABEL = "Change / max |change|\n(within measure)"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font-subsetting chatter


def apply_publication_style() -> None:
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def load_behavior_pivot() -> dict[tuple[str, str], pd.DataFrame]:
    """{(task, metric): DataFrame indexed by subject_id with columns {1, 2}}."""
    df = pd.read_csv(BEHAV_LONG_CSV)
    df = df[df["subject_id"].isin(LONGITUDINAL_SUBJECTS)
            & df["task_order"].isin([1, 2])].copy()
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for task, metric, _, _ in ALL_METRICS:
        sub = df[(df.task == task) & (df.metric == metric)]
        pivot = sub.pivot_table(
            index="subject_id", columns="task_order", values="value"
        ) * DISPLAY_SCALE.get((task, metric), 1.0)
        pivot = pivot.reindex(LONGITUDINAL_SUBJECTS)
        for col in (1, 2):
            if col not in pivot.columns:
                pivot[col] = np.nan
        out[(task, metric)] = pivot
    return out


def check_against_summary(pivots: dict[tuple[str, str], pd.DataFrame]) -> None:
    """Fail loudly if any metric's (n_pairs, n_unchanged, n_improved),
    recomputed from behavior_long.csv, differs from consistency_summary.csv.
    n_pairs counts the pairs that changed; ties are n_unchanged."""
    summary = pd.read_csv(CONSISTENCY_CSV).set_index("metric")
    mismatches = 0
    for task, metric, _label, direction in ALL_METRICS:
        pivot = pivots[(task, metric)]
        deltas = (pivot[2] - pivot[1]).dropna()
        n_unch = int((deltas == 0).sum())
        changed = deltas[deltas != 0]
        n_imp = int((changed < 0).sum() if direction == "lower"
                    else (changed > 0).sum())
        got = (len(changed), n_unch, n_imp)
        row = summary.loc[f"{task}_{metric}"]
        live = (int(row["n_pairs"]), int(row["n_unchanged"]), int(row["n_improved"]))
        if got != live:
            log.error(f"CHECK MISMATCH: {task}/{metric}: "
                      f"behavior_long={got} consistency_summary={live}")
            mismatches += 1
        else:
            log.info(f"  {task}/{metric}: {n_imp} of {len(changed)} changed "
                     f"pairs improved, {n_unch} unchanged")
    if mismatches:
        raise SystemExit(
            f"Cross-check failed: {mismatches} metric(s) disagree with "
            "consistency_summary.csv"
        )
    log.info(f"Cross-check passed: {len(ALL_METRICS)}/{len(ALL_METRICS)} metrics "
             "match consistency_summary.csv")


# --------------------------------------------------------------------------
# Slopegraph panels (one file per task)
# --------------------------------------------------------------------------


def _draw_slope_panel(ax, pivot: pd.DataFrame, label: str, direction: str
                      ) -> None:
    any_plotted = False
    pairs = {s: (pivot.loc[s, 1], pivot.loc[s, 2]) for s in LONGITUDINAL_SUBJECTS
             if pd.notna(pivot.loc[s, 1]) and pd.notna(pivot.loc[s, 2])}
    if pairs:
        allv = [v for pr in pairs.values() for v in pr]
        span = (max(allv) - min(allv)) or 1.0
        dx_pre = dodge_offsets({s: pr[0] for s, pr in pairs.items()}, span)
        dx_post = dodge_offsets({s: pr[1] for s, pr in pairs.items()}, span)
    for subj in LONGITUDINAL_SUBJECTS:
        if subj not in pairs:
            continue
        pre, post = pairs[subj]
        ax.plot(
            [0 + dx_pre[subj], 1 + dx_post[subj]], [pre, post],
            color=SUBJECT_LINE_COLOR, marker=SUBJECT_MARKER[subj],
            linewidth=1.3, markersize=4.8,
            markerfacecolor=SUBJECT_MARKER_COLOR,
            markeredgecolor="white", markeredgewidth=0.5,
        )
        any_plotted = True
    if not any_plotted:
        ax.set_visible(False)
        return
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Time 2"], fontsize=8, rotation=0)
    ax.set_xlim(-0.22, 1.22)
    ax.tick_params(axis="y", labelsize=8)
    caret = "↓" if direction == "lower" else "↑"
    ax.set_title(f"{caret} {label}", fontsize=8.5, pad=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.locator_params(axis="y", nbins=3)


def draw_slopegraph_grid(metrics: list[tuple[str, str, str, str]],
                         pivots: dict[tuple[str, str], pd.DataFrame],
                         out_path: Path,
                         ) -> None:
    """3x3 grid of mini-panels. Hides any cells beyond len(metrics)."""
    fig = plt.figure(figsize=(6.4, 5.0))
    gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0.62, wspace=0.45,
                          left=0.07, right=0.99, top=0.94, bottom=0.06)
    for i in range(9):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        if i < len(metrics):
            task, metric, label, direction = metrics[i]
            _draw_slope_panel(ax, pivots[(task, metric)], label, direction)
        else:
            ax.set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  wrote {out_path.with_suffix('.pdf').name} + .png")


# --------------------------------------------------------------------------
# Heatmap panels (one file per task)
# --------------------------------------------------------------------------


def compute_change_matrix(metrics: list[tuple[str, str, str, str]],
                          pivots: dict[tuple[str, str], pd.DataFrame],
                          subjects: list[str],
                          ) -> np.ndarray:
    """Direction-aligned Delta / max|Delta| within each metric.

    Shape: (n_metrics, len(subjects)). Positive = improvement. NaN where pre
    or post is missing.
    """
    n_metrics = len(metrics)
    n_subjects = len(subjects)
    Z = np.full((n_metrics, n_subjects), np.nan)
    for i, (task, metric, _label, direction) in enumerate(metrics):
        pivot = pivots[(task, metric)]
        for j, subj in enumerate(subjects):
            pre = pivot.loc[subj, 1]
            post = pivot.loc[subj, 2]
            if pd.isna(pre) or pd.isna(post):
                continue
            delta = post - pre
            aligned = -delta if direction == "lower" else delta
            Z[i, j] = aligned
        row = Z[i, :]
        if np.isfinite(row).any():
            mx = np.nanmax(np.abs(row))
            if mx > 0:
                Z[i, :] = row / mx
    return Z


def draw_heatmap_task(metrics: list[tuple[str, str, str, str]],
                      pivots: dict[tuple[str, str], pd.DataFrame],
                      out_path: Path,
                      subjects: list[str],
                      ) -> None:
    """Heatmap for one task block. No colorbar (rendered separately).

    subjects controls the columns drawn: pass a task-specific subject list
    so a subject excluded from that task (e.g. S7, whose baseline Stroop is
    at/below chance) does not appear as a hatched "missing" column -- it is
    absent from the panel entirely, not present-but-missing.
    """
    Z = compute_change_matrix(metrics, pivots, subjects)
    n_metrics, n_subjects = Z.shape

    # ~0.42 in per row, plus space for left labels + top tick labels.
    row_in = 0.42
    fig_h = n_metrics * row_in + 0.6
    fig = plt.figure(figsize=(3.6, fig_h))
    ax = fig.add_axes((0.42, 0.04, 0.55, 1.0 - 0.04 - 0.18 / fig_h - 0.02))

    cmap = plt.get_cmap(CMAP_CHANGE).copy()
    cmap.set_bad(color="#dddddd")
    Zm = np.ma.masked_invalid(Z)
    ax.imshow(Zm, cmap=cmap, vmin=-1.0, vmax=1.0,
              aspect="auto", interpolation="none")

    # Hatch NaN cells
    for i in range(n_metrics):
        for j in range(n_subjects):
            if not np.isfinite(Z[i, j]):
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        hatch="////", fill=False,
                        edgecolor="#888888", linewidth=0.0,
                    )
                )

    # Row labels: caret + measure name
    row_labels = []
    for _task, _metric, label, direction in metrics:
        caret = "↓" if direction == "lower" else "↑"
        row_labels.append(f"{caret} {label}")
    ax.set_yticks(np.arange(n_metrics))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(axis="y", length=0)

    # Subject IDs at the top, in this panel's subject order.
    ax.set_xticks(np.arange(n_subjects))
    ax.set_xticklabels(
        [subject_short(s) for s in subjects],
        fontsize=8.5, rotation=0,
    )
    ax.tick_params(axis="x", length=0, top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  wrote {out_path.with_suffix('.pdf').name} + .png")


# --------------------------------------------------------------------------
# Standalone subject legend
# --------------------------------------------------------------------------


def draw_subject_legend(out_path: Path) -> None:
    fig = plt.figure(figsize=(5.2, 0.45))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    handles = [
        Line2D([0], [0],
               color=SUBJECT_LINE_COLOR, marker=SUBJECT_MARKER[s],
               markerfacecolor=SUBJECT_MARKER_COLOR,
               markeredgecolor="white", markeredgewidth=0.6,
               markersize=7, linewidth=1.6,
               label=subject_short(s))
        for s in LONGITUDINAL_SUBJECTS
    ]
    ax.legend(handles=handles, loc="center", ncol=5, frameon=False,
              fontsize=9, handletextpad=0.5, columnspacing=1.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  wrote {out_path.with_suffix('.pdf').name} + .png")


# --------------------------------------------------------------------------
# Standalone change colorbar
# --------------------------------------------------------------------------


def draw_change_colorbar(out_path: Path) -> None:
    fig = plt.figure(figsize=(1.75, 3.4))
    ax = fig.add_axes((0.36, 0.08, 0.11, 0.84))
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(CMAP_CHANGE), norm=norm, orientation="vertical"
    )
    cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.set_ticklabels(["-1", "-0.5", "0", "+0.5", "+1"], fontsize=8)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    # Quantity on the left, endpoint words on the right: the heatmap files
    # have no titles, and the direction of "improvement" differs by measure.
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(CHANGE_LABEL, fontsize=10)
    endpoint = {"improvement": (0.97, plt.get_cmap(CMAP_CHANGE)(0.95)),
                "no change": (0.50, "#444444"),
                "worsening": (0.03, plt.get_cmap(CMAP_CHANGE)(0.05))}
    for word, (y, colour) in endpoint.items():
        fig.text(0.68, y, word, fontsize=8, ha="left", va="center",
                 color=colour)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  wrote {out_path.with_suffix('.pdf').name} + .png")


def clear_panels(out_dir: Path, prefix: str) -> None:
    """Delete earlier renders, so a renamed or no-longer-drawn panel does
    not survive next to the current ones."""
    for p in list(out_dir.glob(f"{prefix}_*.pdf")) + list(out_dir.glob(f"{prefix}_*.png")):
        p.unlink()


def main() -> None:
    global FIG_OUT, FIG_SUPP_OUT, BEHAV_LONG_CSV, CONSISTENCY_CSV
    output_dir = env_dir("OUTPUT_DIR")
    figures = figure_inputs(output_dir, REQUIRED)
    FIG_OUT = figures / "main" / "Figure3_behavior"
    FIG_SUPP_OUT = figures / "supplementary" / "FigureS2_behavior_change_heatmaps"
    BEHAV_LONG_CSV = output_dir / BEHAV_LONG
    CONSISTENCY_CSV = output_dir / CONSISTENCY

    apply_publication_style()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIG_SUPP_OUT.mkdir(parents=True, exist_ok=True)
    log.info(f"Loading {BEHAV_LONG_CSV}")
    pivots = load_behavior_pivot()
    check_against_summary(pivots)
    clear_panels(FIG_OUT, "fig5")
    clear_panels(FIG_SUPP_OUT, "figS_behavior_change_heatmap")
    log.info(f"Writing panels to {FIG_OUT} and {FIG_SUPP_OUT}")
    # Figure 3: slopegraphs and legend only.
    draw_slopegraph_grid(STROOP_METRICS, pivots,
                         FIG_OUT / "fig5_a_slopegraph_stroop")
    draw_slopegraph_grid(WCST_METRICS, pivots,
                         FIG_OUT / "fig5_b_slopegraph_wcst")
    draw_subject_legend(FIG_OUT / "fig5_c_subject_legend")
    # Supplementary figure: the change heatmaps and their colour bar.
    stem = f"{FIG_SUPP_OUT}/figS_behavior_change_heatmap"
    draw_heatmap_task(STROOP_METRICS, pivots, Path(f"{stem}_a_stroop"),
                      subjects=STROOP_SUBJECTS)
    draw_heatmap_task(WCST_METRICS, pivots, Path(f"{stem}_b_wcst"),
                      subjects=LONGITUDINAL_SUBJECTS)
    draw_change_colorbar(Path(f"{stem}_c_colorbar"))
    log.info("Done.")


if __name__ == "__main__":
    main()
