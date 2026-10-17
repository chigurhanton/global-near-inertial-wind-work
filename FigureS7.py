# -*- coding: utf-8 -*-
"""Reproduce `amplitude_ratio_sites_rmse.png` from the bundled CSV.

The figure summarises 45 near-inertial current deployments at five mooring
sites, comparing the raw slab model with the deep-learning (DL) correction:

  (a) model / observed near-inertial amplitude ratio, one column per site
  (b) RMSE of the 10-day segment amplitude, same layout

Within a site the slab box sits on the left and the DL box on the right. Boxes
are the interquartile range, the thick bar is the median, whiskers reach the most
extreme deployment, and every deployment is drawn as a point. Sites alternate
white / light-grey columns so each site reads as one unit.

This script is self-contained: it needs only this folder and the CSV in
`data/`. No project tree, no model fields, no network.

    python amplitude_ratio_figure.py
    python amplitude_ratio_figure.py --csv other.csv --out my_figure.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
})

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "segment_amplitude_summary_all.csv"
DEFAULT_OUT = HERE / "plots" / "FigureS7.png"

SLAB = "#1f6fb4"
DL = "#c0392b"
SITE_ORDER = ["PAPA", "WHOTS", "KEO", "NTAS", "STRATUS"]
DISPLAY = {"PAPA": "PAPA", "WHOTS": "WHOTS", "KEO": "KEO",
           "NTAS": "NTAS", "STRATUS": "Stratus"}
REQUIRED = {"site", "ratio_slab", "ratio_dl", "rmse_slab", "rmse_dl"}

BOX_GAP = 0.19     # half the distance between the slab and DL boxes of a site
BOX_HALF = 0.17    # half width of a box, in site units
CAP_HALF = 0.06    # half width of a whisker cap
BAND_GREY = "#f0f0f0"


def vbox(ax, values, x, colour):
    """IQR box with median, whiskers to the extremes, and every deployment."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    lo, hi = float(values.min()), float(values.max())
    ax.add_patch(Rectangle((x - BOX_HALF, q1), 2 * BOX_HALF, q3 - q1,
                           facecolor=colour, alpha=0.18, edgecolor=colour,
                           lw=0.7, zorder=2))
    ax.plot([x - BOX_HALF, x + BOX_HALF], [median, median],
            color=colour, lw=1.6, zorder=4)
    ax.plot([x, x], [lo, q1], color=colour, lw=0.8, zorder=3)
    ax.plot([x, x], [q3, hi], color=colour, lw=0.8, zorder=3)
    ax.plot([x - CAP_HALF, x + CAP_HALF], [lo, lo], color=colour, lw=0.8, zorder=3)
    ax.plot([x - CAP_HALF, x + CAP_HALF], [hi, hi], color=colour, lw=0.8, zorder=3)
    ax.plot(np.full(len(values), x), values, "o", ms=2.6, color=colour,
            markeredgewidth=0, zorder=5)
    return float(median)


def pair(ax, slab, dl, x):
    """One site: slab box on the left, DL box on the right."""
    vbox(ax, slab, x - BOX_GAP, SLAB)
    vbox(ax, dl, x + BOX_GAP, DL)


def build_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise SystemExit("%s is missing required columns: %s"
                         % (csv_path, ", ".join(sorted(missing))))
    frame["site"] = frame["site"].replace({"Stratus": "STRATUS"})
    return frame


def make_figure(frame: pd.DataFrame, out: Path, dpi: int = 400) -> None:
    sites = [s for s in SITE_ORDER if s in set(frame.site)]
    if not sites:
        raise SystemExit("no recognised site in the CSV "
                         "(expected any of %s)" % ", ".join(SITE_ORDER))
    labels = ["%s (%d)" % (DISPLAY[s], int((frame.site == s).sum())) for s in sites]
    x = {s: i + 1 for i, s in enumerate(sites)}

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.6))

    for s in sites:
        sub = frame[frame.site == s]
        pair(axes[0], sub.ratio_slab.to_numpy(), sub.ratio_dl.to_numpy(), x[s])
        pair(axes[1], sub.rmse_slab.to_numpy(), sub.rmse_dl.to_numpy(), x[s])

    # Column striping: each site owns one strip, so the two boxes and the
    # tick label are visually tied together without touching the boxes.
    for ax in axes:
        for i, s in enumerate(sites):
            if i % 2 == 0:
                ax.axvspan(x[s] - 0.5, x[s] + 0.5, color=BAND_GREY, zorder=0)

    axes[0].axhline(1.0, color="0.35", ls="--", lw=0.8, zorder=1)
    axes[0].text(len(sites) + 0.48, 1.005, "obs", color="0.35", fontsize=7.0,
                 ha="right", va="bottom")
    axes[0].set_ylim(0.3, 1.25)
    axes[0].set_ylabel("Model / observed near-inertial amplitude")
    axes[1].set_ylim(0.0, 0.125)
    axes[1].set_ylabel("Amplitude RMSE (m s$^{-1}$)")

    for ax, panel in zip(axes, ("a", "b")):
        ax.set_xticks([x[s] for s in sites])
        ax.set_xticklabels(labels)
        # Exactly the extent of the stripes: the bands span x +/- 0.5, so
        # anything wider would leave an unstripped sliver at either end.
        ax.set_xlim(0.5, len(sites) + 0.5)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(panel, loc="left", fontweight="bold")

    handles = [plt.Line2D([], [], marker="s", ls="", ms=4, color=SLAB, label="slab"),
               plt.Line2D([], [], marker="s", ls="", ms=4, color=DL, label="DL")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, frameon=False, handletextpad=0.4, columnspacing=2.0)
    fig.tight_layout(h_pad=1.8, rect=(0, 0, 1, 0.955))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def summarise(frame: pd.DataFrame) -> None:
    table = frame.groupby("site").agg(
        n=("key", "size") if "key" in frame.columns else ("site", "size"),
        ratio_slab=("ratio_slab", "median"), ratio_dl=("ratio_dl", "median"),
        rmse_slab=("rmse_slab", "median"), rmse_dl=("rmse_dl", "median")
    ).round(3)
    print(table.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="input table (default: data/segment_amplitude_summary_all.csv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output PNG (default: output/amplitude_ratio_sites_rmse.png)")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    frame = build_frame(args.csv)
    make_figure(frame, args.out, args.dpi)
    print("deployments: %d   sites: %d" % (len(frame), frame.site.nunique()))
    summarise(frame)
    print("saved: %s" % args.out.resolve())


if __name__ == "__main__":
    main()
