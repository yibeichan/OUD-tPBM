#!/usr/bin/env python3
"""
Figure 2 and Supplementary Figure S1: longitudinal self-report trajectories
for the participants with a second timepoint.

One panel file per assessment (baseline -> time 2, one line per
participant). Lines are neutral grey and each participant has a marker
shape of his own; Figure 3 uses the same shapes.

Second timepoint: each participant's time-2 point is drawn at the session
`10_longitudinal/unified_ef/second_timepoint_selection.csv` selects (the
later session with the most usable levels, ties to post), for every
assessment, so the trajectories use the same session per participant as
Figures 3 and S3-S4, with one exception for the questionnaires below.

Questionnaire pairing: when a participant's selected second-timepoint
visit has no PHQ-9, GAD-7 or BSCS value but another post-baseline visit
(post or followup) has one, that questionnaire's time-2 value comes from
the post-baseline visit closest to the selected one (the earlier visit on a
tie). Each questionnaire is paired on its own; BRIEF-A is never
substituted. The pipeline's descriptive longitudinal stage applies the same
rule to PHQ-9 and GAD-7 (timepoint_selection.apply_second_timepoint_questionnaires);
it does not load BSCS, so BSCS is paired in this script only.

BRIEF-A administrations that fail any validity scale are dropped.

Output files (letters follow the caption order):
  OUTPUT_DIR/figures/main/Figure2_self_report/
    fig6_a_brief_gec    BRIEF-A Global Executive Composite T-score, with the
                        T = 65 clinical cutoff as a dashed line
    fig6_b_phq9         Patient Health Questionnaire-9 total score
    fig6_c_gad7         General Anxiety Disorder-7 total score
    fig6_d_bscs         3-item Brief Substance Craving Scale total
                        (`bscs_total`, 0-12)
    fig6_e_subject_legend
  OUTPUT_DIR/figures/supplementary/FigureS1_BRIEF-A_subscales/
    figS_brief_subscales_a_inhibit, _b_shift, _c_wm (BRIEF-A T-scores, same
    axis and cutoff line as fig6_a), _d_subject_legend

check_against_summary() recomputes (n_pairs changed, n_unchanged,
n_improved) per measure and stops if it differs from
10_longitudinal/consistency_summary.csv, for every measure that table has.
BSCS has no row there, so it is not cross-checked, and the log says so.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D

from _common import env_dir, figure_inputs
from brief_validity import valid_administration_mask

load_dotenv()

ASSESS = "01_assessments"
SECOND_TIMEPOINT = "10_longitudinal/unified_ef/second_timepoint_selection.csv"
CONSISTENCY = "10_longitudinal/consistency_summary.csv"
# Inputs relative to OUTPUT_DIR.
REQUIRED = [
    f"{ASSESS}/scores/gad7_scores.csv",
    f"{ASSESS}/scores/phq9_scores.csv",
    f"{ASSESS}/scores/craving_scores.csv",
    *(f"{ASSESS}/t-scores/executive_function_{tp}_subscale_t-scores.csv"
      for tp in ("baseline", "post", "followup")),
    SECOND_TIMEPOINT,
    CONSISTENCY,
]
# Set in main() from OUTPUT_DIR.
FIG_OUT: Path
FIG_SUPP_OUT: Path
SCORES_DIR: Path
TSCORES_DIR: Path
SECOND_TIMEPOINT_CSV: Path
CONSISTENCY_CSV: Path
BRIEF_CUTOFF_T = 65

# Subject IDs are S-codes throughout (assessment CSVs included).
LONGITUDINAL_SUBJECTS = ["S1", "S3", "S5", "S7", "S10"]

# Subject encoding: every line is the same neutral grey and each participant
# has a marker shape of his own, so colour does not encode participant. Same
# constants in fig3_figS2_behavior.py.
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
        cl = sorted(cl, key=LONGITUDINAL_SUBJECTS.index)
        for k, subj in enumerate(cl):
            out[subj] = (k - (len(cl) - 1) / 2) * DODGE_STEP
    return out


def subject_short(subject_id: str) -> str:
    """Identity: subject IDs are already S-codes."""
    return subject_id


def load_second_timepoint_labels() -> dict[str, str]:
    """{subject: "post" | "followup"}, from the second-timepoint selection
    (later session with most usable levels, ties to post). Same table the
    other longitudinal figures use, so all show the same session per
    subject."""
    if not SECOND_TIMEPOINT_CSV.exists():
        raise SystemExit(f"Missing {SECOND_TIMEPOINT_CSV}; cannot resolve second timepoints")
    df = pd.read_csv(SECOND_TIMEPOINT_CSV)
    order_to_label = {2: "post", 3: "followup"}
    return {
        str(row.subject_id): order_to_label[int(row.t2_order)]
        for row in df.itertuples()
    }


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


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


# Visits a second timepoint can come from, by visit order.
POST_BASELINE_ORDER = {"post": 2, "followup": 3}


def _value_at(rows: pd.DataFrame, metric_col: str, tp_label: str) -> float | None:
    sel = rows[rows["timepoint"] == tp_label]
    if not sel.empty and pd.notna(sel.iloc[0][metric_col]):
        return float(sel.iloc[0][metric_col])
    return None


def _resolve_second_timepoint(per_subject: pd.DataFrame, metric_col: str,
                               second_tp_labels: dict[str, str],
                               *, questionnaire: bool = False,
                               ) -> dict[str, dict[str, float | None]]:
    """{subject: {"baseline": v_or_None, "post_resolved": v_or_None,
    "second_timepoint": "post"|"followup", "source_timepoint": label|None,
    "used_followup": bool}}.

    Reads baseline plus whichever timepoint second_timepoint_selection.csv
    designates as that subject's second timepoint -- not "post if present,
    else followup." With ``questionnaire=True`` (PHQ-9, GAD-7, BSCS), a
    subject whose designated timepoint has no value takes it from the
    post-baseline visit closest to the designated one, the earlier on a
    tie; BRIEF-A is loaded with ``questionnaire=False`` and is never
    substituted. A subject still without a value gets
    post_resolved=None and draws no line. ``source_timepoint`` is the visit
    the second-timepoint value was recorded at.
    """
    out: dict[str, dict] = {}
    for subj in LONGITUDINAL_SUBJECTS:
        rows = per_subject[per_subject["subject"] == subj]
        tp_label = second_tp_labels[subj]
        rec: dict = {"baseline": _value_at(rows, metric_col, "baseline"),
                     "post_resolved": _value_at(rows, metric_col, tp_label),
                     "second_timepoint": tp_label, "source_timepoint": None,
                     "used_followup": tp_label == "followup"}
        if rec["post_resolved"] is not None:
            rec["source_timepoint"] = tp_label
        elif questionnaire:
            target = POST_BASELINE_ORDER[tp_label]
            for label in sorted(
                (lab for lab in POST_BASELINE_ORDER if lab != tp_label),
                key=lambda lab: (abs(POST_BASELINE_ORDER[lab] - target),
                                 POST_BASELINE_ORDER[lab]),
            ):
                value = _value_at(rows, metric_col, label)
                if value is not None:
                    rec["post_resolved"], rec["source_timepoint"] = value, label
                    break
        out[subj] = rec
    return out


def load_scores_assessment(filename: str, second_tp_labels: dict[str, str],
                            metric_col: str = "total_score"
                            ) -> dict[str, dict]:
    """Load gad7/phq9 long-format scores csv (questionnaire pairing applies)."""
    df = pd.read_csv(SCORES_DIR / filename)
    return _resolve_second_timepoint(df, metric_col, second_tp_labels,
                                     questionnaire=True)


def load_bscs(second_tp_labels: dict[str, str]) -> dict[str, dict]:
    """3-item BSCS total (`bscs_total`, 0-12) from the craving scores."""
    fpath = SCORES_DIR / "craving_scores.csv"
    df = pd.read_csv(fpath)
    if "bscs_total" not in df.columns:
        raise SystemExit(f"{fpath} has no bscs_total column; re-run stage 01 "
                         "(scripts/assessments/04_analyze_craving.py)")
    vals = df["bscs_total"].dropna()
    if not vals.between(0, 12).all():
        raise SystemExit("bscs_total outside 0-12")
    return _resolve_second_timepoint(df, "bscs_total", second_tp_labels,
                                     questionnaire=True)


def check_against_summary(measures: dict[str, dict[str, dict]]) -> None:
    """Stop if paired-change counts differ from consistency_summary.csv,
    for every measure that table has."""
    summary = pd.read_csv(CONSISTENCY_CSV).set_index("metric")
    bad = 0
    for key, values in measures.items():
        deltas = [rec["post_resolved"] - rec["baseline"] for rec in values.values()
                  if rec["baseline"] is not None and rec["post_resolved"] is not None]
        n_unch = sum(d == 0 for d in deltas)
        changed = [d for d in deltas if d != 0]
        got = (len(changed), n_unch, sum(d < 0 for d in changed))
        live = None
        if key in summary.index:
            row = summary.loc[key]
            live = (int(row["n_pairs"]), int(row["n_unchanged"]), int(row["n_improved"]))
        if live is not None and live != got:
            log.error(f"CHECK MISMATCH {key}: computed={got} "
                      f"consistency_summary={live}")
            bad += 1
        else:
            log.info(f"  {key}: {got[2]} of {got[0]} changed pairs improved, "
                     f"{got[1]} unchanged"
                     + ("" if live is not None else
                        " (not cross-checked: consistency_summary.csv has no row for it)"))
    if bad:
        raise SystemExit(f"Cross-check failed for {bad} measure(s)")


def load_brief_t_scores(metric_col: str, second_tp_labels: dict[str, str]
                         ) -> dict[str, dict]:
    """Pull BRIEF-A T-scores for one subscale across timepoints.

    The BRIEF-A T-scores live in three per-timepoint files (baseline, post,
    followup). We stitch them into a single subject -> timepoint -> value
    map, nulling out administrations that fail validity (main analysis
    variant setting: exclude_invalid_brief). metric_col is one of: Inhibit_t,
    Shift_t, Working Memory_t, GEC_t.
    """
    frames = []
    for tp in ("baseline", "post", "followup"):
        fp = TSCORES_DIR / f"executive_function_{tp}_subscale_t-scores.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        invalid = ~valid_administration_mask(df)
        df = df[["subject", metric_col]].copy()
        if invalid.any():
            log.info(f"  excluding invalid BRIEF-A administration(s) at {tp}: "
                     f"{df.loc[invalid, 'subject'].tolist()}")
        df.loc[invalid, metric_col] = np.nan
        df["timepoint"] = tp
        df = df.rename(columns={metric_col: "value"})
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    return _resolve_second_timepoint(long, "value", second_tp_labels)


# --------------------------------------------------------------------------
# Panel drawing
# --------------------------------------------------------------------------


def draw_assessment_panel(values: dict[str, dict], out_path: Path,
                           y_lim: tuple[float, float] | None = None,
                           y_label: str | None = None,
                           cutoff: float | None = None,
                           ) -> None:
    """One panel: baseline -> second-timepoint slopegraph for one assessment."""
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes((0.20, 0.16, 0.76, 0.78))
    if cutoff is not None:
        ax.axhline(cutoff, color="#555555", linestyle="--", linewidth=0.9,
                   zorder=0)

    pairs = {s: (values[s]["baseline"], values[s]["post_resolved"])
             for s in LONGITUDINAL_SUBJECTS
             if values[s]["baseline"] is not None
             and values[s]["post_resolved"] is not None}
    if y_lim is not None:
        span = y_lim[1] - y_lim[0]
    else:
        allv = [v for pr in pairs.values() for v in pr]
        span = (max(allv) - min(allv)) or 1.0
    dx_pre = dodge_offsets({s: pr[0] for s, pr in pairs.items()}, span)
    dx_post = dodge_offsets({s: pr[1] for s, pr in pairs.items()}, span)
    for subj in LONGITUDINAL_SUBJECTS:
        if subj not in pairs:
            continue
        pre, post = pairs[subj]
        marker = SUBJECT_MARKER[subj]
        ax.plot(
            [0 + dx_pre[subj], 1 + dx_post[subj]], [pre, post],
            color=SUBJECT_LINE_COLOR, marker=marker,
            linewidth=1.4, markersize=6.0,
            markerfacecolor=SUBJECT_MARKER_COLOR,
            markeredgecolor="white", markeredgewidth=0.6,
            clip_on=False,  # scores of 0 sit on the x-axis; keep markers whole
            zorder=3,  # above the axis spine, so a line along 0 stays visible
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Time 2"], fontsize=9, rotation=0)
    ax.set_xlim(-0.22, 1.22)

    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=10)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  wrote {out_path.with_suffix('.pdf').name} + .png")


# --------------------------------------------------------------------------
# Shared subject legend (matches Figure 3; replicated in each folder for
# self-contained assembly)
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
# Provenance
# --------------------------------------------------------------------------


def log_subject_provenance(name: str, values: dict[str, dict]) -> None:
    """Print baseline, second-timepoint value, and which timepoint per subject."""
    log.info(f"--- {name} ---")
    for subj in LONGITUDINAL_SUBJECTS:
        rec = values[subj]
        src = rec["source_timepoint"]
        flag = f" ({src.upper()})" if src and src != "post" else ""
        if src and src != rec["second_timepoint"]:
            flag += f" (from {src}, selected {rec['second_timepoint']}; questionnaire pairing)"
        log.info(
            f"  {subject_short(subj)}: baseline={rec['baseline']}, "
            f"post_resolved={rec['post_resolved']}{flag}"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def clear_panels(out_dir: Path, prefix: str) -> None:
    """Delete earlier renders, so a renamed or no-longer-drawn panel does
    not survive next to the current ones."""
    for p in list(out_dir.glob(f"{prefix}_*.pdf")) + list(out_dir.glob(f"{prefix}_*.png")):
        p.unlink()


def main() -> None:
    global FIG_OUT, FIG_SUPP_OUT, SCORES_DIR, TSCORES_DIR
    global SECOND_TIMEPOINT_CSV, CONSISTENCY_CSV
    output_dir = env_dir("OUTPUT_DIR")
    figures = figure_inputs(output_dir, REQUIRED)
    FIG_OUT = figures / "main" / "Figure2_self_report"
    FIG_SUPP_OUT = figures / "supplementary" / "FigureS1_BRIEF-A_subscales"
    SCORES_DIR = output_dir / ASSESS / "scores"
    TSCORES_DIR = output_dir / ASSESS / "t-scores"
    SECOND_TIMEPOINT_CSV = output_dir / SECOND_TIMEPOINT
    CONSISTENCY_CSV = output_dir / CONSISTENCY

    apply_publication_style()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIG_SUPP_OUT.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading from {output_dir / ASSESS}")

    second_tp_labels = load_second_timepoint_labels()
    log.info(f"  second timepoint per subject: {second_tp_labels}")

    # Clinical scales
    gad7 = load_scores_assessment("gad7_scores.csv", second_tp_labels)
    phq9 = load_scores_assessment("phq9_scores.csv", second_tp_labels)
    bscs = load_bscs(second_tp_labels)

    # BRIEF-A subscales (T-scores; lower = better, normative T~50, range ~30-100)
    brief_gec = load_brief_t_scores("GEC_t", second_tp_labels)
    brief_wm = load_brief_t_scores("Working Memory_t", second_tp_labels)
    brief_shift = load_brief_t_scores("Shift_t", second_tp_labels)
    brief_inhibit = load_brief_t_scores("Inhibit_t", second_tp_labels)

    log_subject_provenance("GAD-7", gad7)
    log_subject_provenance("PHQ-9", phq9)
    log_subject_provenance("BSCS", bscs)
    log_subject_provenance("BRIEF-A GEC", brief_gec)
    log_subject_provenance("BRIEF-A Working Memory", brief_wm)
    log_subject_provenance("BRIEF-A Shift", brief_shift)
    log_subject_provenance("BRIEF-A Inhibit", brief_inhibit)

    check_against_summary({
        "gad7_total_score": gad7, "phq9_total_score": phq9, "bscs_total": bscs,
        "executive_function_gec_t": brief_gec,
        "executive_function_working_memory_t": brief_wm,
        "executive_function_shift_t": brief_shift,
        "executive_function_inhibit_t": brief_inhibit,
    })

    clear_panels(FIG_OUT, "fig6")
    clear_panels(FIG_SUPP_OUT, "figS_brief_subscales")
    log.info(f"Writing panels to {FIG_OUT} and {FIG_SUPP_OUT}")

    # BRIEF-A panels share one y-range so the GEC panel (Figure 2) and the
    # subscale panels (supplement) are directly comparable. Normative mean
    # T~50; clinical cutoff 65. The measure is named on the axis.
    brief_ylim = (30, 100)

    # Figure 2: GEC, PHQ-9, GAD-7, BSCS, legend.
    draw_assessment_panel(brief_gec, FIG_OUT / "fig6_a_brief_gec",
                          y_lim=brief_ylim, y_label="BRIEF-A GEC (T)",
                          cutoff=BRIEF_CUTOFF_T)
    draw_assessment_panel(phq9, FIG_OUT / "fig6_b_phq9",
                          y_lim=(0, 15), y_label="PHQ-9 total score")
    draw_assessment_panel(gad7, FIG_OUT / "fig6_c_gad7",
                          y_lim=(0, 15), y_label="GAD-7 total score")
    draw_assessment_panel(bscs, FIG_OUT / "fig6_d_bscs",
                          y_lim=(0, 12), y_label="BSCS total score (3 items)")
    draw_subject_legend(FIG_OUT / "fig6_e_subject_legend")

    # Supplementary figure: the three BRIEF-A subscales the pathways use.
    stem = FIG_SUPP_OUT / "figS_brief_subscales"
    draw_assessment_panel(brief_inhibit, Path(f"{stem}_a_inhibit"),
                          y_lim=brief_ylim, y_label="BRIEF-A Inhibit (T)",
                          cutoff=BRIEF_CUTOFF_T)
    draw_assessment_panel(brief_shift, Path(f"{stem}_b_shift"),
                          y_lim=brief_ylim, y_label="BRIEF-A Shift (T)",
                          cutoff=BRIEF_CUTOFF_T)
    draw_assessment_panel(brief_wm, Path(f"{stem}_c_wm"),
                          y_lim=brief_ylim, y_label="BRIEF-A Working Memory (T)",
                          cutoff=BRIEF_CUTOFF_T)
    draw_subject_legend(Path(f"{stem}_d_subject_legend"))

    log.info("Done.")


if __name__ == "__main__":
    main()
