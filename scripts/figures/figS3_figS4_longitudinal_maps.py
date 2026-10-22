#!/usr/bin/env python3
"""
Supplementary Figures S3 and S4: longitudinal pre vs post fNIRS β topomaps
for the participants with a second timepoint.

Per contrast, this produces:
  - per participant, {pre, post} per-channel β panels
  - 1 group directional-consistency panel showing
        signed_consistency = (n_pos - n_neg) / n_total  ∈ [-1, +1]
    where n_pos / n_neg / n_total count participants whose post − pre Δβ at
    that channel is positive / negative / non-NaN. Channels covered by fewer
    than 2 participants are masked out. Drawn as one dot per channel
    coloured by its value, without interpolation, with a stepped colour bar
    listing only the values attainable for the n involved.

β maps are interpolated linearly between channels; the head outline of
every map is schematic: centred on a sphere fitted to the task's plotted
channels and scaled so every plotted channel of the figure lies inside it.
β is in µM (SNIRF HbO data unit "uM", read unscaled by MNE).

Sessions: a session at or below chance accuracy is excluded when the
variant sets exclude_low_accuracy (04b_session_info/low_accuracy_sessions.csv),
so a participant whose baseline session of a task is excluded has no panels
for that task.

Second timepoint:
    Per participant, the "post" timepoint is whichever of task_order 2
    (post) or 3 (followup) the second-timepoint selection chose (the later
    session with the most usable levels, ties to post). This is read from
    10_longitudinal/unified_ef/second_timepoint_selection.csv rather than
    hard-coded, so a re-run that changes the selection is picked up
    automatically.

Channels:
    When the run's analysis_variant.json sets channels = "long" (main),
    each session's short channels (< 10 mm, listed per session in
    04b_session_info/short_channels_{task}.csv) are dropped before
    plotting, as scripts/brain/roi_utils.py::read_glm_contrasts does in
    the pipeline. The 03_glm CSVs are first-level output and still contain
    the short channels, which entered the GLM only as nuisance regressors.

Contrasts:
    Figure S3 carries the four contrasts the H1-H4 pathways test
    (MAIN_CONTRASTS). The remaining three are rendered the same way into
    Figure S4.

Inputs: a ``main`` run tree (OUTPUT_DIR), plus BASE_DIR/data/{task}_task_info.csv
and the SNIRF recordings under BASE_DIR/data/{stroop,wcst}/ for the montage.

Outputs:
    OUTPUT_DIR/figures/supplementary/FigureS3_longitudinal_maps_pathway_contrasts/
    OUTPUT_DIR/figures/supplementary/FigureS4_longitudinal_maps_other_contrasts/
        fig4_{letter}_{task}_{contrast}_S{N}_{pre|post}.{pdf,png}
        fig4_{letter}_{task}_{contrast}_group_consistency.{pdf,png}
        fig4_{letter}_{task}_{contrast}_beta_colorbar.{pdf,png}
        fig4_{letter}_{task}_{contrast}_consistency_colorbar.{pdf,png}
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import mne
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))
from channel_names import build_canonical_montage  # noqa: E402
from _common import env_dir, figure_inputs  # noqa: E402
from analysis_variant import load_variant  # noqa: E402

load_dotenv()

SECOND_TIMEPOINT = "10_longitudinal/unified_ef/second_timepoint_selection.csv"
# Inputs relative to OUTPUT_DIR. The 04b_session_info tables are read only
# when the variant asks for the rule they carry; main asks for all three.
REQUIRED = [
    SECOND_TIMEPOINT,
    "03_glm/stroop",
    "03_glm/wcst",
    "04b_session_info/low_accuracy_sessions.csv",
    "04b_session_info/wcst_condition_counts.csv",
    "04b_session_info/short_channels_stroop.csv",
    "04b_session_info/short_channels_wcst.csv",
]
# Set in main() from BASE_DIR and OUTPUT_DIR.
BASE_DIR: Path
OUTPUT_DIR: Path
FIG_OUT: Path
FIG_SUPP_OUT: Path
GLM_DIR: Path
SECOND_TIMEPOINT_CSV: Path

LONGITUDINAL_SUBJECTS = ["S1", "S3", "S5", "S7", "S10"]
DEFAULT_POST_ORDER = 2


def load_post_order_override() -> dict[str, int]:
    """{subject: t2_order} for subjects whose selected second timepoint is not
    task_order=2 (post), read from the second-timepoint selection (later
    session with the most usable levels, ties to post).
    """
    df = pd.read_csv(SECOND_TIMEPOINT_CSV)
    return {
        str(row.subject_id): int(row.t2_order)
        for row in df.itertuples()
        if int(row.t2_order) != DEFAULT_POST_ORDER
    }

STROOP_CONTRASTS = ["stroop_effect", "mixing_cost", "incongruent_vs_pure"]
WCST_CONTRASTS = [
    "correct_vs_incorrect",
    "change_vs_regular",
    "learning",
    "duration_effect_correct",
]

# Contrasts in the published figure, keyed to the pathway each one tests.
MAIN_CONTRASTS = {
    "stroop_effect": "H1",
    "change_vs_regular": "H2",
    "learning": "H3",
    "mixing_cost": "H4",
}

# WCST contrasts the main analysis treats as missing (not just noisy) for a
# session with zero regular_incorrect trials -- mirrors
# scripts/brain/roi_utils.py::MAINTENANCE_CONTRASTS, applied there in read_glm_contrasts() when the run's
# analysis_variant.json sets mask_wcst_without_regular_incorrect=true and
# scripts/brain/session_info.py::session_lacks_regular_incorrect() is true
# for that session. duration_effect_correct is deliberately excluded from
# this list -- the pipeline does not mask it.
MAINTENANCE_CONTRASTS = ("correct_vs_incorrect", "change_vs_regular", "learning")

# Panel letter per contrast (filename only; no letter is drawn). Figure S3
# in pathway order H1-H4; Figure S4 has the other three contrasts. Each
# contrast carries its own β and consistency colour bars.
PANEL_LETTER = {
    "stroop_effect": "a",
    "change_vs_regular": "b",
    "learning": "c",
    "mixing_cost": "d",
    "incongruent_vs_pure": "a",
    "correct_vs_incorrect": "b",
    "duration_effect_correct": "c",
}

# Contrast names as printed on the colour bars.
CONTRAST_LABEL = {
    "stroop_effect": "Stroop effect",
    "mixing_cost": "mixing cost",
    "incongruent_vs_pure": "incongruent vs pure",
    "correct_vs_incorrect": "correct vs incorrect",
    "change_vs_regular": "change vs regular",
    "learning": "learning",
    "duration_effect_correct": "duration effect (correct)",
}

CMAP_BETA = "RdBu_r"
CMAP_CONSISTENCY = "RdBu_r"
IMAGE_INTERP = "linear"
# Consistency dots (size chosen so the dense central channels overlap
# little at the 2.6-in panel): size (pt^2) and the thin edge
# that keeps a 0 (white) dot visible on the white page.
DOT_SIZE = 2.2
DOT_EDGE = "#8c8c8c"
# Head outline: fitted centre, radius
# scaled so the farthest plotted channel of the figure sits just inside.
HEAD_MARGIN = 1.03

# Per-contrast vlim = p95 of |β| over all included sessions for that contrast.
# Per-task averaging mixed contrasts with β scales an order of magnitude
# apart (e.g. WCST duration_effect_correct vs learning), washing out the
# small-β contrasts. Saturating ~5% of channels keeps the bulk of the
# distribution visually distinguishable.
BETA_PERCENTILE = 95

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


def load_task_info(task: str) -> pd.DataFrame:
    fpath = Path(BASE_DIR) / "data" / f"{task}_task_info.csv"
    return pd.read_csv(fpath, dtype={"task_id": str})


def load_excluded_sessions(variant: dict,
                           output_dir: str | Path | None = None) -> set[tuple[str, str, int]]:
    """{(task, subject, task_order)} at or below chance accuracy, excluded
    when the variant sets `exclude_low_accuracy`.

    Empty when `exclude_low_accuracy` is false (e.g. supp_low_accuracy),
    matching the pipeline's session_exclusions.excluded_sessions: the CSV
    logs the rule's result on every tree, so it is not the exclusion list
    by itself. SystemExit if the variant excludes but the CSV is missing:
    silently excluding nothing would put a below-chance session back in.
    """
    if "exclude_low_accuracy" not in variant:
        raise SystemExit(
            "analysis_variant.json has no 'exclude_low_accuracy' key; "
            "cannot determine whether below-chance sessions are excluded"
        )
    if not variant["exclude_low_accuracy"]:
        return set()
    root = Path(output_dir or OUTPUT_DIR)
    fpath = root / "04b_session_info" / "low_accuracy_sessions.csv"
    if not fpath.exists():
        raise SystemExit(
            f"Missing {fpath}, but analysis_variant.json sets "
            "exclude_low_accuracy = true; refusing to run without the "
            "exclusion list (a below-chance session would re-enter the figure)"
        )
    df = pd.read_csv(fpath)
    return set(zip(df["task"], df["subject"], df["task_order"]))


def load_wcst_condition_counts(variant: dict) -> pd.DataFrame | None:
    """Read 04b_session_info/wcst_condition_counts.csv, or None when the
    variant does not mask WCST maintenance contrasts at all (no session
    lookup is needed in that case). SystemExit if the flag is true but the
    file is missing.
    """
    if not variant["mask_wcst_without_regular_incorrect"]:
        return None
    fpath = OUTPUT_DIR / "04b_session_info" / "wcst_condition_counts.csv"
    if not fpath.exists():
        raise SystemExit(
            f"Missing {fpath}; mask_wcst_without_regular_incorrect=true "
            "requires session trial counts to apply the mask"
        )
    return pd.read_csv(fpath, dtype={"session_id": str})


def load_wcst_masked_sessions(condition_counts: pd.DataFrame | None) -> frozenset[str]:
    """WCST session_ids (task_id hashes) with zero regular_incorrect
    trials, for logging only. Empty when condition_counts
    is None (the variant doesn't mask)."""
    if condition_counts is None:
        return frozenset()
    return frozenset(
        condition_counts.loc[condition_counts["regular_incorrect"] == 0, "session_id"]
    )


def session_lacks_regular_incorrect(session_hash: str,
                                     condition_counts: pd.DataFrame | None,
                                     ) -> bool:
    """Whether one WCST session has zero regular_incorrect trials.

    Mirrors scripts/brain/session_info.py::session_lacks_regular_incorrect:
    condition_counts is None only when
    the variant doesn't mask at all (never lacks, by construction).
    Otherwise every session this script looks up must be present in
    wcst_condition_counts.csv -- a session absent from that file is a
    SystemExit here, mirroring the pipeline's KeyError, never silently
    treated as "not masked."
    """
    if condition_counts is None:
        return False
    session_df = condition_counts[condition_counts["session_id"] == session_hash]
    if session_df.empty:
        raise SystemExit(
            f"session {session_hash!r} not found in wcst_condition_counts.csv; "
            "cannot determine whether its regular_incorrect count is zero "
            "(mask_wcst_without_regular_incorrect=true requires every WCST "
            "session this figure uses to be listed there)"
        )
    return bool((session_df["regular_incorrect"] == 0).all())


_SHORT_CHANNEL_TABLES: dict[str, pd.DataFrame] = {}


def short_channels(task: str, session_hash: str) -> frozenset[str]:
    """Canonical names (S#_D#) of one session's short channels.

    Mirrors scripts/brain/session_info.py::short_channels: a session
    absent from short_channels_{task}.csv is an
    error, never treated as having no short channels.
    """
    if task not in _SHORT_CHANNEL_TABLES:
        fpath = OUTPUT_DIR / "04b_session_info" / f"short_channels_{task}.csv"
        if not fpath.exists():
            raise SystemExit(f"Missing {fpath}; channels = long requires the "
                             "per-session short-channel list")
        _SHORT_CHANNEL_TABLES[task] = pd.read_csv(fpath, dtype={"session_id": str})
    df = _SHORT_CHANNEL_TABLES[task]
    sel = df[df["session_id"] == session_hash]
    if sel.empty:
        raise SystemExit(f"session {session_hash!r} not found in "
                         f"short_channels_{task}.csv")
    return frozenset(sel.loc[sel["is_short"].astype(bool), "channel"])


def find_session_hash(task_info: pd.DataFrame, subject: str,
                      task_order: int) -> str | None:
    sel = task_info[(task_info["subj_id"] == subject)
                    & (task_info["task_order"] == task_order)]
    if sel.empty:
        return None
    return str(sel["task_id"].iloc[0])


def load_session_betas(task: str, subject: str, session_hash: str,
                       contrast: str,
                       condition_counts: pd.DataFrame | None = None,
                       long_only: bool = True,
                       ) -> dict[str, float]:
    """Return {base_channel_name: β} for HbO channels and the given contrast.

    Mirrors roi_utils.read_glm_contrasts's WCST maintenance-contrast mask:
    the 03_glm CSVs are the pipeline's raw, pre-mask output, so a session
    with zero regular_incorrect trials still has real (fit-without-those-
    trials) numbers for MAINTENANCE_CONTRASTS. When the variant masks such
    sessions and this one lacks regular_incorrect trials, treat this
    contrast as missing here too (empty dict) rather than plotting those
    numbers. session_lacks_regular_incorrect raises if session_hash isn't
    in condition_counts at all, so an unlisted session is never silently
    treated as unmasked. With long_only, the session's short channels are
    dropped (variant channels = "long").
    """
    if (task == "wcst" and contrast in MAINTENANCE_CONTRASTS
            and session_lacks_regular_incorrect(session_hash, condition_counts)):
        return {}
    fpath = GLM_DIR / task / f"{subject}_{session_hash}_glm_contrasts.csv"
    df = pd.read_csv(fpath)
    df = df[(df["chromophore"] == "hbo") & (df["contrast"] == contrast)]
    # Canonical name from Source/Detector, as read_glm_contrasts builds it.
    channels = ("S" + df["Source"].astype(int).astype(str)
                + "_D" + df["Detector"].astype(int).astype(str))
    drop = short_channels(task, session_hash) if long_only else frozenset()
    out: dict[str, float] = {}
    for ch, eff in zip(channels, df["effect"]):
        if pd.isna(eff) or ch in drop:
            continue
        out[ch] = float(eff)
    return out


def load_snirf_montage(task: str) -> tuple[mne.Info, dict[str, int]]:
    """Canonical union HbO montage across every SNIRF recording for the task.

    fNIRS channel counts vary per session, and a single recording's
    positions lack channels other sessions recorded. The montage is the
    union of all
    recordings under canonical channel names (see
    scripts/brain/channel_names.py), so a channel absent from one session's
    03_glm output is simply not plotted for that session/timepoint, rather
    than being unrepresentable in the montage at all.
    """
    snirf_dir = Path(BASE_DIR) / "data" / task
    snirf_files = sorted(snirf_dir.glob("*.snirf"))
    if not snirf_files:
        raise FileNotFoundError(f"No SNIRF files found in {snirf_dir}")
    info, ch_mapping = build_canonical_montage(snirf_files)
    log.info(f"  master montage: {len(ch_mapping)} HbO channels (union of "
             f"{len(snirf_files)} recordings)")
    return info, ch_mapping


def fit_montage_sphere(info: mne.Info, channels: list[int]) -> np.ndarray:
    """Least-squares sphere through the positions of the plotted channels,
    as MNE's (x, y, z, radius) `sphere` argument.

    Same fit as fig1_glm_tmaps.py: MNE's default sphere (origin 0,
    radius 95 mm) is not fitted to this probe, and sphere="auto" cannot be
    used because the SNIRF files carry no head digitisation in head
    coordinates (LPA, RPA and one more point only). One sphere per task,
    through every channel drawn in any of the task's maps.
    """
    locs = np.array([info["chs"][i]["loc"][:3] for i in channels])
    a = np.c_[2 * locs, np.ones(len(locs))]
    coef, *_ = np.linalg.lstsq(a, (locs ** 2).sum(axis=1), rcond=None)
    origin = coef[:3]
    radius = float(np.sqrt(coef[3] + origin @ origin))
    resid = np.linalg.norm(locs - origin, axis=1) - radius
    log.info(f"  fitted head sphere ({len(locs)} ch): origin "
             f"({origin[0] * 1e3:.1f}, {origin[1] * 1e3:.1f}, "
             f"{origin[2] * 1e3:.1f}) mm, radius {radius * 1e3:.1f} mm, "
             f"residual SD {resid.std() * 1e3:.1f} mm")
    return np.r_[origin, radius]


def topomap_positions(info: mne.Info, sphere: np.ndarray) -> np.ndarray:
    """2-D positions MNE's plot_topomap uses for these channels."""
    from mne.channels.layout import _find_topomap_coords  # no public API
    return _find_topomap_coords(info, picks=np.arange(len(info["ch_names"])),
                                sphere=sphere)


def head_distances(info: mne.Info, sphere: np.ndarray) -> np.ndarray:
    """Distance of each channel's 2-D position from the outline centre.
    MNE projects each channel by its own distance from the sphere origin,
    so the positions do not depend on the sphere radius."""
    pos = topomap_positions(info, sphere)
    return np.hypot(pos[:, 0] - sphere[0], pos[:, 1] - sphere[1])


def n_outside_head(info: mne.Info, sphere: np.ndarray) -> int:
    """Channels whose 2-D position falls outside the head circle."""
    return int((head_distances(info, sphere) > sphere[3]).sum())


def check_interp_range(im, values: np.ndarray, name: str) -> None:
    """Stop if the interpolated image leaves the range of the channel values
    (MNE's default cubic interpolation overshoots it; see IMAGE_INTERP)."""
    img = np.asarray(im.get_array(), dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    tol = 1e-6 * max(abs(lo), abs(hi), 1.0)
    if np.nanmin(img) < lo - tol or np.nanmax(img) > hi + tol:
        raise SystemExit(
            f"{name}: interpolated map spans [{np.nanmin(img):.2f}, "
            f"{np.nanmax(img):.2f}], outside the channel range "
            f"[{lo:.2f}, {hi:.2f}]"
        )


def _values_on_montage(values_by_ch: dict[str, float], info: mne.Info,
                       ch_mapping: dict[str, int]
                       ) -> tuple[mne.Info, np.ndarray] | None:
    """(info restricted to the channels with a value, their values)."""
    n_ch = len(info["ch_names"])
    plot_values = np.full(n_ch, np.nan)
    for ch_name, val in values_by_ch.items():
        idx = ch_mapping.get(ch_name)
        if idx is None or val is None or np.isnan(val):
            continue
        plot_values[idx] = val
    valid_idx = np.where(~np.isnan(plot_values))[0]
    if not valid_idx.size:
        return None
    return mne.pick_info(info, valid_idx), plot_values[valid_idx]


def render_value_topomap(values_by_ch: dict[str, float], info: mne.Info,
                          ch_mapping: dict[str, int], vlim: float,
                          cmap: str, out_path: Path,
                          sphere: np.ndarray) -> tuple[int, int]:
    """Interpolated β map; returns (channels drawn, channels outside the
    head outline)."""
    picked = _values_on_montage(values_by_ch, info, ch_mapping)
    if picked is None:
        log.warning(f"  no overlapping channels for {out_path.name}")
        return 0, 0
    info_subset, plot_subset = picked

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    im, _ = mne.viz.plot_topomap(
        plot_subset,
        info_subset,
        axes=ax,
        cmap=cmap,
        vlim=(-vlim, vlim),
        sphere=sphere,
        sensors=False,
        contours=6,
        extrapolate="local",
        # Linear, not MNE's default cubic: cubic interpolation between
        # neighbouring channels can overshoot the channel values by an
        # order of magnitude, so saturated blobs and contour levels would
        # show values no channel has. Linear stays within the channel range.
        image_interp=IMAGE_INTERP,
        show=False,
    )
    ax.set_title("")
    check_interp_range(im, plot_subset, out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return len(plot_subset), n_outside_head(info_subset, sphere)


def consistency_levels(coverage_ns: set[int]) -> list[float]:
    """Values (n_pos - n_neg) / n can take for the coverages n in the map,
    with every Δβ non-zero (n_pos + n_neg = n): (2k - n) / n, k = 0..n."""
    return sorted({(2 * k - n) / n for n in coverage_ns for k in range(n + 1)})


def level_colour(value: float):
    """Colour of one consistency value: RdBu_r at the value's position on
    [-1, 1], so a value has the same colour in every panel."""
    return plt.get_cmap(CMAP_CONSISTENCY)((value + 1.0) / 2.0)


def render_consistency_dots(values_by_ch: dict[str, float], info: mne.Info,
                            ch_mapping: dict[str, int], levels: list[float],
                            out_path: Path, sphere: np.ndarray
                            ) -> tuple[int, int]:
    """One dot per channel at its topomap position, coloured by its value;
    no interpolation. The head outline and framing come
    from an invisible plot_topomap call on the same channels, so the panel
    lines up with the β maps."""
    picked = _values_on_montage(values_by_ch, info, ch_mapping)
    if picked is None:
        log.warning(f"  no overlapping channels for {out_path.name}")
        return 0, 0
    info_subset, values = picked
    off_level = [v for v in values if not np.isclose(levels, v).any()]
    if off_level:
        raise SystemExit(f"{out_path.stem}: {len(off_level)} channel value(s) "
                         f"not among the attainable levels (e.g. {off_level[0]:.3f}); "
                         "a Δβ of exactly 0 would do this")
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    clear = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 0.0)])
    mne.viz.plot_topomap(values, info_subset, axes=ax, cmap=clear,
                         vlim=(-1.0, 1.0), sphere=sphere, sensors=False,
                         contours=0, extrapolate="local",
                         image_interp=IMAGE_INTERP, show=False)
    pos = topomap_positions(info_subset, sphere)
    order = np.argsort(np.abs(values), kind="stable")  # strongest on top
    ax.scatter(pos[order, 0], pos[order, 1], s=DOT_SIZE,
               c=[level_colour(v) for v in values[order]],
               edgecolors=DOT_EDGE, linewidths=0.15, zorder=5)
    for line in ax.lines:  # head, nose and ears above the frontal dots
        line.set_zorder(6)
    ax.set_title("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return len(values), n_outside_head(info_subset, sphere)


def level_label(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    text = str(abs(Fraction(value).limit_denominator(12)))
    return ("+" if value > 0 else "\u2212") + text


def render_stepped_colorbar(levels: list[float], label: str,
                            out_path: Path) -> None:
    """Vertical colour bar with one block per attainable consistency value
    (vertical so fraction labels such as -1/2 and -1/3 do not collide)."""
    fig = plt.figure(figsize=(1.45, max(2.2, 0.24 * len(levels))))
    ax = fig.add_axes((0.50, 0.03, 0.16, 0.94))
    cmap = mpl.colors.ListedColormap([level_colour(v) for v in levels])
    norm = mpl.colors.BoundaryNorm(np.arange(len(levels) + 1) - 0.5, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation="vertical")
    cb.set_ticks(np.arange(len(levels)))
    cb.set_ticklabels([level_label(v) for v in levels])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(label, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def render_colorbar(vlim: float, label: str, cmap: str, out_path: Path,
                     ticks: list[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(2.6, 0.75))
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=norm,
                                   orientation="horizontal")
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    if ticks is None:
        ticks = list(np.linspace(-vlim, vlim, 5))
    cb.set_ticks(ticks)
    labels = []
    for t in ticks:
        labels.append("0" if abs(t) < 1e-9 else f"{t:+.2g}")
    cb.set_ticklabels(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def resolve_session_pairs(task: str, task_info: pd.DataFrame,
                           post_order_override: dict[str, int],
                           excluded_sessions: set[tuple[str, str, int]],
                           ) -> tuple[dict[str, dict], list[str]]:
    """Per subject, look up pre (task_order=1) and post hashes.

    post_order_override gives the selected second-timepoint task_order for
    subjects whose second timepoint is followup rather than post (from
    second_timepoint_selection.csv). Sessions at or below chance accuracy
    (excluded_sessions) are treated as missing.
    Subjects without a qualifying pair are skipped with a warning.
    """
    pairs: dict[str, dict] = {}
    used_followup_for_post: list[str] = []
    for subj in LONGITUDINAL_SUBJECTS:
        pre_hash = find_session_hash(task_info, subj, 1)
        if (task, subj, 1) in excluded_sessions:
            log.warning(f"  {subj}: excluding baseline {task} "
                        "(at/below chance accuracy)")
            pre_hash = None
        post_order = post_order_override.get(subj, DEFAULT_POST_ORDER)
        post_hash = find_session_hash(task_info, subj, post_order)
        if (task, subj, post_order) in excluded_sessions:
            log.warning(f"  {subj}: excluding {task} task_order={post_order} "
                        "(at/below chance accuracy)")
            post_hash = None
        if pre_hash is None or post_hash is None:
            log.warning(f"  {subj}: missing pre or post (pre={pre_hash}, "
                        f"post[task_order={post_order}]={post_hash}) — skipping")
            continue
        pairs[subj] = {"pre_hash": pre_hash, "post_hash": post_hash,
                       "post_order": post_order}
        if post_order != DEFAULT_POST_ORDER:
            used_followup_for_post.append(subj)
    return pairs, used_followup_for_post


def collect_subject_betas(task: str, contrasts: list[str],
                           pairs: dict[str, dict],
                           condition_counts: pd.DataFrame | None,
                           long_only: bool,
                           ) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    """{subj: {contrast: {'pre': {ch: β}, 'post': {ch: β}}}}, plus a list of
    (subject, contrast, 'pre'|'post') triples that came back masked (empty)
    because that session has zero regular_incorrect trials."""
    out: dict[str, dict] = {}
    masked_log: list[tuple[str, str, str]] = []
    for subj, p in pairs.items():
        out[subj] = {}
        for contrast in contrasts:
            session_hash = {"pre": p["pre_hash"], "post": p["post_hash"]}
            betas = {
                tp: load_session_betas(task, subj, session_hash[tp], contrast,
                                       condition_counts, long_only)
                for tp in ("pre", "post")
            }
            out[subj][contrast] = betas
            for tp in ("pre", "post"):
                if (task == "wcst" and contrast in MAINTENANCE_CONTRASTS
                        and session_lacks_regular_incorrect(session_hash[tp],
                                                             condition_counts)):
                    masked_log.append((subj, contrast, tp))
    return out, masked_log


def compute_beta_vlim_per_contrast(subject_data: dict[str, dict],
                                     contrasts: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for contrast in contrasts:
        all_betas: list[float] = []
        for by_contrast in subject_data.values():
            by_tp = by_contrast.get(contrast, {})
            for ch_dict in by_tp.values():
                all_betas.extend(ch_dict.values())
        if not all_betas:
            out[contrast] = 1.0
        else:
            out[contrast] = float(np.percentile(np.abs(all_betas),
                                                BETA_PERCENTILE))
    return out


def compute_group_consistency(subject_data: dict[str, dict], contrast: str,
                               ch_mapping: dict[str, int]
                               ) -> tuple[dict[str, float], dict[str, int]]:
    consistency: dict[str, float] = {}
    coverage: dict[str, int] = {}
    for ch_base in ch_mapping.keys():
        deltas: list[float] = []
        for by_contrast in subject_data.values():
            pre_b = by_contrast[contrast]["pre"].get(ch_base)
            post_b = by_contrast[contrast]["post"].get(ch_base)
            if pre_b is None or post_b is None:
                continue
            if np.isnan(pre_b) or np.isnan(post_b):
                continue
            deltas.append(post_b - pre_b)
        if len(deltas) < 2:
            continue
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        n_total = len(deltas)
        consistency[ch_base] = (n_pos - n_neg) / n_total
        coverage[ch_base] = n_total
    return consistency, coverage


def panel_dir(contrast: str) -> Path:
    """Main-figure directory for a hypothesis contrast, else the supplement."""
    return FIG_OUT if contrast in MAIN_CONTRASTS else FIG_SUPP_OUT


def panel_stem(task: str, contrast: str) -> Path:
    """Directory plus filename prefix, fig4_<letter>_<task>_<contrast>."""
    return panel_dir(contrast) / f"fig4_{PANEL_LETTER[contrast]}_{task}_{contrast}"


def figure_of(contrast: str) -> str:
    return "main" if contrast in MAIN_CONTRASTS else "supplement"


def load_task(task: str, contrasts: list[str],
              post_order_override: dict[str, int],
              excluded_sessions: set[tuple[str, str, int]],
              condition_counts: pd.DataFrame | None,
              long_only: bool) -> dict:
    """Sessions, β values and the fitted outline centre for one task, plus
    the farthest plotted channel per figure (for the outline radius)."""
    log.info(f"=== {task.upper()} ===")
    info, ch_mapping = load_snirf_montage(task)
    task_info = load_task_info(task)

    pairs, used_followup = resolve_session_pairs(
        task, task_info, post_order_override, excluded_sessions
    )
    if used_followup:
        log.info(f"  using task_order=3 (followup) as post for: {used_followup}")
    # Sort by canonical subject order, not lexicographically (unpadded
    # S-codes sort "S10" before "S3").
    ordered_subjects = [s for s in LONGITUDINAL_SUBJECTS if s in pairs]
    log.info(f"  {len(pairs)} subjects with valid pre/post pairs: {ordered_subjects}")

    subject_data, masked_log = collect_subject_betas(task, contrasts, pairs,
                                                      condition_counts, long_only)
    for subj in ordered_subjects:
        p = pairs[subj]
        n_pre = len(subject_data[subj][contrasts[0]]["pre"])
        n_post = len(subject_data[subj][contrasts[0]]["post"])
        log.info(f"  {subj}: pre {p['pre_hash']} ({n_pre} ch), "
                 f"T2 task_order={p['post_order']} {p['post_hash']} ({n_post} ch)"
                 f"{' [long channels only]' if long_only else ''}")
    if masked_log:
        log.info(f"  masked (0 regular_incorrect, treated as missing): {masked_log}")
    plotted = sorted({ch_mapping[ch]
                      for by_contrast in subject_data.values()
                      for by_tp in by_contrast.values()
                      for ch_dict in by_tp.values()
                      for ch in ch_dict if ch in ch_mapping})
    sphere = fit_montage_sphere(info, plotted)
    r_max: dict[str, float] = {}
    for fig in ("main", "supplement"):
        idx = sorted({ch_mapping[ch]
                      for by_contrast in subject_data.values()
                      for contrast, by_tp in by_contrast.items()
                      if figure_of(contrast) == fig
                      for ch_dict in by_tp.values()
                      for ch in ch_dict if ch in ch_mapping})
        r_max[fig] = float(head_distances(mne.pick_info(info, idx), sphere).max())
        log.info(f"  {fig}: farthest plotted channel {r_max[fig] * 1e3:.1f} mm "
                 f"from the centre (fitted radius {sphere[3] * 1e3:.1f} mm)")
    return {"task": task, "contrasts": contrasts, "info": info,
            "ch_mapping": ch_mapping, "pairs": pairs,
            "ordered_subjects": ordered_subjects,
            "used_followup": used_followup, "subject_data": subject_data,
            "masked_log": masked_log, "sphere": sphere, "r_max": r_max}


def build_task(state: dict, radius: dict[str, float]) -> dict:
    """Render one task's panels with one outline radius per figure."""
    task, contrasts = state["task"], state["contrasts"]
    info, ch_mapping = state["info"], state["ch_mapping"]
    pairs, ordered_subjects = state["pairs"], state["ordered_subjects"]
    subject_data, masked_log = state["subject_data"], state["masked_log"]
    used_followup = state["used_followup"]
    log.info(f"=== {task.upper()} (render) ===")

    def sphere_for(contrast: str) -> np.ndarray:
        return np.r_[state["sphere"][:3], radius[figure_of(contrast)]]

    vlim_per_contrast = compute_beta_vlim_per_contrast(subject_data, contrasts)
    for c, v in vlim_per_contrast.items():
        log.info(f"  β vlim {c} (p{BETA_PERCENTILE} of |β|): ±{v:.3f}")

    # Individual subject panels (β at pre and post) — per-contrast vlim
    outside_log: list[tuple[str, int, int]] = []
    for subj, by_contrast in subject_data.items():
        for contrast, by_tp in by_contrast.items():
            v = vlim_per_contrast[contrast]
            for tp, ch_dict in by_tp.items():
                out = panel_stem(task, contrast).with_name(
                    f"{panel_stem(task, contrast).name}_{subj}_{tp}")
                n, n_out = render_value_topomap(ch_dict, info, ch_mapping,
                                                 v, CMAP_BETA, out,
                                                 sphere_for(contrast))
                if n:
                    outside_log.append((out.name, n, n_out))

    # Per-contrast β colorbar
    for contrast in contrasts:
        v = vlim_per_contrast[contrast]
        stem = panel_stem(task, contrast)
        render_colorbar(v, f"β (µM), HbO\n{CONTRAST_LABEL[contrast]}", CMAP_BETA,
                        stem.with_name(f"{stem.name}_beta_colorbar"))

    # Group directional-consistency panel per contrast
    coverage_summary = {}
    for contrast in contrasts:
        consistency, coverage = compute_group_consistency(
            subject_data, contrast, ch_mapping
        )
        stem = panel_stem(task, contrast)
        out = stem.with_name(f"{stem.name}_group_consistency")
        levels = consistency_levels(set(coverage.values()))
        n_ch, n_out = render_consistency_dots(consistency, info, ch_mapping,
                                              levels, out, sphere_for(contrast))
        outside_log.append((out.name, n_ch, n_out))
        render_stepped_colorbar(
            levels, "Δβ sign consistency\n(time 2 − baseline)",
            stem.with_name(f"{stem.name}_consistency_colorbar"))
        cov = list(coverage.values())
        cov_summary = {
            "n_channels_displayed": n_ch,
            "min_subjects": int(min(cov)) if cov else 0,
            "max_subjects": int(max(cov)) if cov else 0,
            "mean_subjects": float(np.mean(cov)) if cov else 0.0,
            "n_values": sorted(set(cov)),
            "levels": levels,
        }
        coverage_summary[contrast] = cov_summary
        log.info(f"  {contrast} consistency: {n_ch} ch displayed, "
                 f"coverage min/mean/max = {cov_summary['min_subjects']}/"
                 f"{cov_summary['mean_subjects']:.2f}/"
                 f"{cov_summary['max_subjects']} subjects")

    outside = [(name, o) for name, _n, o in outside_log if o]
    if outside:
        raise SystemExit(f"{task}: channels outside the head outline in "
                         f"{len(outside)} map(s), e.g. {outside[0]}")
    n_ch_all = [n for _name, n, _o in outside_log]
    log.info(f"  0 channels outside the head outline in all "
             f"{len(outside_log)} maps ({min(n_ch_all)}-{max(n_ch_all)} "
             "channels drawn)")
    return {
        "sphere": state["sphere"],
        "radius": radius,
        "n_maps": len(outside_log),
        "n_drawn": (min(n_ch_all), max(n_ch_all)),
        "pairs": {s: pairs[s] for s in ordered_subjects},
        "vlim_per_contrast": vlim_per_contrast,
        "n_subjects": len(pairs),
        "subjects": ordered_subjects,
        "used_followup_for_post": used_followup,
        "coverage_summary": coverage_summary,
        "masked_log": masked_log,
    }


def clear_panels(out_dir: Path, prefix: str) -> None:
    """Delete earlier renders, so a renamed or no-longer-drawn panel does
    not survive next to the current ones."""
    for p in list(out_dir.glob(f"{prefix}_*.pdf")) + list(out_dir.glob(f"{prefix}_*.png")):
        p.unlink()


def main() -> None:
    global BASE_DIR, OUTPUT_DIR, FIG_OUT, FIG_SUPP_OUT, GLM_DIR, SECOND_TIMEPOINT_CSV
    OUTPUT_DIR = env_dir("OUTPUT_DIR")
    BASE_DIR = env_dir("BASE_DIR")
    figures = figure_inputs(OUTPUT_DIR, REQUIRED) / "supplementary"
    FIG_OUT = figures / "FigureS3_longitudinal_maps_pathway_contrasts"
    FIG_SUPP_OUT = figures / "FigureS4_longitudinal_maps_other_contrasts"
    GLM_DIR = OUTPUT_DIR / "03_glm"
    SECOND_TIMEPOINT_CSV = OUTPUT_DIR / SECOND_TIMEPOINT

    apply_publication_style()
    for out_dir in (FIG_OUT, FIG_SUPP_OUT):
        out_dir.mkdir(parents=True, exist_ok=True)
        clear_panels(out_dir, "fig4")

    post_order_override = load_post_order_override()
    log.info(f"  second-timepoint override (subject -> task_order): {post_order_override}")
    variant = asdict(load_variant(OUTPUT_DIR))
    excluded_sessions = load_excluded_sessions(variant)
    log.info(f"  exclude_low_accuracy={variant['exclude_low_accuracy']}; "
             f"excluded (at/below chance) sessions: {sorted(excluded_sessions)}")
    condition_counts = load_wcst_condition_counts(variant)
    masked_sessions = load_wcst_masked_sessions(condition_counts)
    log.info(f"  mask_wcst_without_regular_incorrect={variant['mask_wcst_without_regular_incorrect']}; "
             f"masked WCST session(s): {sorted(masked_sessions)}")

    if variant.get("channels") not in ("long", "all"):
        raise SystemExit("analysis_variant.json 'channels' must be 'long' or "
                         f"'all', got {variant.get('channels')!r}")
    long_only = variant["channels"] == "long"
    log.info(f"  channels={variant['channels']}")

    stroop_state = load_task("stroop", STROOP_CONTRASTS, post_order_override,
                              excluded_sessions, None, long_only)
    wcst_state = load_task("wcst", WCST_CONTRASTS, post_order_override,
                            excluded_sessions, condition_counts, long_only)
    # One outline radius per figure (Figure S3 = "main", Figure S4 =
    # "supplement"), shared by its Stroop and WCST panels and just enclosing
    # the farthest plotted channel. Each task keeps its own fitted centre.
    radius = {fig: HEAD_MARGIN * max(stroop_state["r_max"][fig],
                                     wcst_state["r_max"][fig])
              for fig in ("main", "supplement")}
    for fig, r in radius.items():
        log.info(f"{fig} head-outline radius: {r * 1e3:.1f} mm "
                 f"(farthest channel x {HEAD_MARGIN})")
    build_task(stroop_state, radius)
    build_task(wcst_state, radius)
    log.info(f"All panels written under {FIG_OUT} and {FIG_SUPP_OUT}")


if __name__ == "__main__":
    main()
