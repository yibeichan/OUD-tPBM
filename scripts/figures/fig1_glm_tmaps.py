#!/usr/bin/env python3
"""
Figure 1: channel-level group t-statistic topomaps for Stroop and WCST.

Produces one panel file per contrast (no titles, no panel labels, no per-panel
colorbars) for manual assembly into the composite figure, plus one shared
colorbar panel per task. Channels surviving FDR (q<0.05) are marked with a
black ring. Values are interpolated linearly between channels and the head
outline is centred on a sphere fitted to the montage and scaled so every
plotted channel lies inside it.

Inputs: the stage-05 channel tables of a ``main`` run tree (OUTPUT_DIR) and
the SNIRF recordings under BASE_DIR/data/{stroop,wcst}/ for the montage.

Outputs:
    OUTPUT_DIR/figures/main/Figure1_Stroop/  (fig1_*: 3 contrasts + colorbar)
    OUTPUT_DIR/figures/main/Figure1_WCST/    (fig2_*: 4 contrasts + colorbar)
"""

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

load_dotenv()
# Set in main() from BASE_DIR and OUTPUT_DIR.
BASE_DIR: Path
CHANNEL_DIR: Path

STROOP_CONTRASTS = ["stroop_effect", "mixing_cost", "incongruent_vs_pure"]
WCST_CONTRASTS = [
    "correct_vs_incorrect",
    "change_vs_regular",
    "learning",
    "duration_effect_correct",
]

# Stage-05 channel tables this figure reads, relative to OUTPUT_DIR.
REQUIRED = [
    f"05_channel_group_analysis/{task}/{task}_{c}_channel_statistics.csv"
    for task, contrasts in (("stroop", STROOP_CONTRASTS), ("wcst", WCST_CONTRASTS))
    for c in contrasts
]

# Output folder and panel-file prefix per task.
TASK_FOLDER = {"stroop": ("Figure1_Stroop", "fig1"), "wcst": ("Figure1_WCST", "fig2")}

# One symmetric color limit for both tasks, so the Stroop and WCST panel sets
# can be assembled into one figure on a single scale. check_stats() stops if
# any |t| exceeds VLIM, so no channel is clipped silently.
VLIM = {"stroop": 4.0, "wcst": 4.0}

CMAP = "RdBu_r"
IMAGE_INTERP = "linear"
# Head outline: fitted centre, radius scaled so the farthest plotted channel
# of the figure sits just inside.
HEAD_MARGIN = 1.03

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


def load_snirf_montage(task: str) -> tuple[mne.Info, dict[str, int]]:
    """Canonical union HbO montage across every SNIRF recording for the task.

    A single recording lacks
    channels other sessions have, so the montage is built from the union
    of all recordings under canonical channel names (see
    scripts/brain/channel_names.py).
    """
    snirf_dir = Path(BASE_DIR) / "data" / task
    snirf_files = sorted(snirf_dir.glob("*.snirf"))
    if not snirf_files:
        raise FileNotFoundError(f"No SNIRF files found in {snirf_dir}")
    info, ch_mapping = build_canonical_montage(snirf_files)
    log.info(f"  montage: {len(ch_mapping)} HbO channels (union of "
             f"{len(snirf_files)} recordings)")
    return info, ch_mapping


def fit_montage_sphere(info: mne.Info, channels: list[int]) -> np.ndarray:
    """Least-squares sphere through the positions of the plotted channels,
    as MNE's (x, y, z, radius) `sphere` argument.

    MNE's default sphere (origin 0, radius 95 mm) is not fitted to this
    probe. sphere="auto" needs head digitisation, and the SNIRF files carry
    only LPA, RPA and one more point, not in head coordinates, so MNE
    cannot fit one. The channel positions lie on (long channels: just
    under) the scalp, so a sphere through the channels the figure draws
    gives the outline the head's centre and radius. Fitting the full
    union montage (short channels and all optodes) moves the centre
    back and up and leaves more plotted channels outside the outline,
    so the plotted set is used.
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


def head_distances(info: mne.Info, sphere: np.ndarray) -> np.ndarray:
    """Distance of each channel's 2-D topomap position from the outline
    centre. MNE projects each channel by its own distance from the sphere
    origin, so the positions do not depend on the sphere radius."""
    from mne.channels.layout import _find_topomap_coords  # no public API
    pos = _find_topomap_coords(info, picks=np.arange(len(info["ch_names"])),
                               sphere=sphere)
    return np.hypot(pos[:, 0] - sphere[0], pos[:, 1] - sphere[1])


def n_outside_head(info: mne.Info, sphere: np.ndarray) -> int:
    """Channels whose 2-D topomap position falls outside the head circle."""
    return int((head_distances(info, sphere) > sphere[3]).sum())


def load_channel_stats(task: str, contrast: str) -> pd.DataFrame:
    fpath = CHANNEL_DIR / task / f"{task}_{contrast}_channel_statistics.csv"
    return pd.read_csv(fpath)


def check_stats(task: str, contrast: str, stats: pd.DataFrame) -> int:
    """Stop if the stage-05 table mixes sample sizes or exceeds VLIM.

    Returns the number of FDR-significant channels for this contrast.
    """
    n_sub = sorted(stats["n_subjects"].unique().tolist())
    if len(n_sub) != 1:
        raise SystemExit(
            f"{task}/{contrast}: channels differ in n_subjects {n_sub}; "
            "stage 05 tests every channel on the same sessions. Check OUTPUT_DIR."
        )
    t_max = float(stats["t_statistic"].abs().max())
    if t_max > VLIM[task]:
        raise SystemExit(f"{task}/{contrast}: max |t| {t_max:.2f} exceeds "
                         f"VLIM {VLIM[task]}; raise VLIM")
    n_sig = int(stats["significant_fdr"].astype(bool).sum())
    log.info(f"  {task}/{contrast}: {len(stats)} ch, n = {n_sub[0]}, "
             f"max |t| = {t_max:.2f}, FDR-sig = {n_sig}, "
             f"min p_FDR = {stats['p_fdr'].min():.3f}")
    return n_sig


def check_interp_range(im, values: np.ndarray, name: str) -> None:
    """Stop if the interpolated image leaves the range of the channel values.

    The colours must show values some channel has; an interpolator that
    overshoots (MNE's default cubic does, by up to 10x here) draws
    activation that is not in the data.
    """
    img = np.asarray(im.get_array(), dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    tol = 1e-6 * max(abs(lo), abs(hi), 1.0)
    if np.nanmin(img) < lo - tol or np.nanmax(img) > hi + tol:
        raise SystemExit(
            f"{name}: interpolated map spans [{np.nanmin(img):.2f}, "
            f"{np.nanmax(img):.2f}], outside the channel range "
            f"[{lo:.2f}, {hi:.2f}]"
        )


def render_topomap(
    stats: pd.DataFrame,
    info: mne.Info,
    ch_mapping: dict[str, int],
    vlim: float,
    out_path: Path,
    sphere: np.ndarray,
) -> None:
    """Render one channel-level t-statistic topomap to `out_path` (.pdf and .png).

    No title, no on-figure labels, no colorbar. FDR-sig channels are marked
    with a black ring via the mask parameter.
    """
    n_ch_info = len(info["ch_names"])
    plot_values = np.full(n_ch_info, np.nan)
    mask = np.zeros(n_ch_info, dtype=bool)

    for _, row in stats.iterrows():
        idx = ch_mapping.get(row["channel"])
        if idx is None:
            continue
        plot_values[idx] = row["t_statistic"]
        if bool(row.get("significant_fdr", False)):
            mask[idx] = True

    valid = ~np.isnan(plot_values)
    if not valid.any():
        log.warning(f"  no overlapping channels for {out_path.stem}")
        return

    valid_idx = np.where(valid)[0]
    info_subset = mne.pick_info(info, valid_idx)
    plot_subset = plot_values[valid_idx]
    mask_subset = mask[valid_idx]

    mask_params = dict(
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        linewidth=1.2,
    )

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    im, _ = mne.viz.plot_topomap(
        plot_subset,
        info_subset,
        axes=ax,
        cmap=CMAP,
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
        mask=mask_subset if mask_subset.any() else None,
        mask_params=mask_params if mask_subset.any() else None,
        show=False,
    )
    ax.set_title("")
    check_interp_range(im, plot_subset, out_path.stem)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    n_marked = int(mask_subset.sum())
    n_out = n_outside_head(info_subset, sphere)
    if n_out:
        raise SystemExit(f"{out_path.stem}: {n_out} channel(s) outside the "
                         "head outline; the outline radius must enclose all")
    log.info(f"  saved {out_path.name}{{.pdf,.png}} ({len(valid_idx)} ch, "
             f"0 outside the head outline, {n_marked} FDR-sig marked)")


def render_colorbar(vlim: float, out_path: Path, label: str = "t-statistic") -> None:
    """Render a horizontal colorbar to its own file."""
    fig, ax = plt.subplots(figsize=(2.6, 0.55))
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(CMAP), norm=norm,
                                   orientation="horizontal")
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    # Symmetric ticks at ±vlim/0
    ticks = np.linspace(-vlim, vlim, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:+.1f}".replace("+0.0", "0").replace("-0.0", "0") for t in ticks])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    log.info(f"  saved {out_path.name}{{.pdf,.png}}")


def load_task(task: str, contrasts: list[str]) -> dict:
    """Montage, stage-05 tables and the fitted outline centre for one task."""
    info, ch_mapping = load_snirf_montage(task)
    all_stats = {c: load_channel_stats(task, c) for c in contrasts}
    plotted = sorted({ch_mapping[ch] for st in all_stats.values()
                      for ch in st["channel"] if ch in ch_mapping})
    sphere = fit_montage_sphere(info, plotted)
    r_max = float(head_distances(mne.pick_info(info, plotted), sphere).max())
    log.info(f"  {task}: farthest plotted channel {r_max * 1e3:.1f} mm from "
             f"the centre (fitted radius {sphere[3] * 1e3:.1f} mm)")
    return {"info": info, "ch_mapping": ch_mapping, "stats": all_stats,
            "sphere": sphere, "r_max": r_max}


def build_figure(task: str, state: dict, contrasts: list[str], fig_dir: Path,
                 prefix: str, panel_letters: list[str], radius: float) -> None:
    log.info(f"=== {task.upper()} -> {fig_dir} ===")
    info, ch_mapping = state["info"], state["ch_mapping"]
    sphere = np.r_[state["sphere"][:3], radius]
    vlim = VLIM[task]

    n_sig_total = 0
    for letter, contrast in zip(panel_letters, contrasts):
        stats = state["stats"][contrast]
        n_sig_total += check_stats(task, contrast, stats)
        out = fig_dir / f"{prefix}_{letter}_{contrast}"
        render_topomap(stats, info, ch_mapping, vlim, out, sphere)
    log.info(f"  {task}: {n_sig_total} FDR-significant channels over "
             f"{len(contrasts)} contrasts")

    cbar_letter = panel_letters[-1]  # next letter after the last contrast panel
    next_letter = chr(ord(cbar_letter) + 1)
    render_colorbar(vlim, fig_dir / f"{prefix}_{next_letter}_colorbar")


def clear_panels(out_dir: Path, prefix: str) -> None:
    """Delete earlier renders, so a renamed or no-longer-drawn panel does
    not survive next to the current ones."""
    for p in list(out_dir.glob(f"{prefix}_*.pdf")) + list(out_dir.glob(f"{prefix}_*.png")):
        p.unlink()


def main() -> None:
    global BASE_DIR, CHANNEL_DIR
    output_dir = env_dir("OUTPUT_DIR")
    BASE_DIR = env_dir("BASE_DIR")
    fig_out = figure_inputs(output_dir, REQUIRED) / "main"
    CHANNEL_DIR = output_dir / "05_channel_group_analysis"

    apply_publication_style()
    for folder, prefix in TASK_FOLDER.values():
        (fig_out / folder).mkdir(parents=True, exist_ok=True)
        clear_panels(fig_out / folder, prefix)

    stroop = load_task("stroop", STROOP_CONTRASTS)
    wcst = load_task("wcst", WCST_CONTRASTS)
    # One outline radius for all Figure 1 panels, just enclosing the
    # farthest plotted channel of either task. Each task keeps its own
    # fitted centre.
    radius = HEAD_MARGIN * max(stroop["r_max"], wcst["r_max"])
    log.info(f"Figure 1 head-outline radius: {radius * 1e3:.1f} mm "
             f"(farthest channel x {HEAD_MARGIN})")
    for task, state, contrasts, letters in (
            ("stroop", stroop, STROOP_CONTRASTS, ["a", "b", "c"]),
            ("wcst", wcst, WCST_CONTRASTS, ["a", "b", "c", "d"])):
        folder, prefix = TASK_FOLDER[task]
        build_figure(task, state, contrasts, fig_out / folder, prefix,
                     panel_letters=letters, radius=radius)

    log.info(f"All panels written under {fig_out}")


if __name__ == "__main__":
    main()
