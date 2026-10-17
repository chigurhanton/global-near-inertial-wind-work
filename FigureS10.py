# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S8 from source data or a reusable cache.

Pointwise diagnostic of slab-model NIW speed bias. The hexbin color denotes
median mixed-layer depth (MLD) within each bin.

Designed for execution as a single cell in a VS Code Jupyter notebook.
"""

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Project paths and analysis settings
# =============================================================================
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "cache"
PLOTS_DIR = PROJECT_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_NPZ = DATA_DIR / "drifter_niw_300h_new.npz"
IDX_GOOD_NPY = DATA_DIR / "idx_good.npy"
CACHE_FILE = CACHE_DIR / "FigS10_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS10.png"

CACHE_VERSION = 1
RECOMPUTE_CACHE = False

# Point-selection thresholds.
SLAB_MIN_SPEED = 0.03  # m s^-1
GT_MIN_SPEED = 0.03    # m s^-1
SMOOTH_C = 0.03        # m s^-1; used only for SMOOTHED_RATIO
EPS = 1e-5

# Diagnostic metric.
# Options: RATIO, SMOOTHED_RATIO, SMAPE, ABS_ERROR
METRIC_TYPE = "RATIO"

# Plot controls.
X_LOG = False
XLIM = (0.0, 1.2)
YLIM = None
GRIDSIZE = 120
MINCNT = 5
NBINS = 60


# =============================================================================
# 2. Figure style
# =============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
})


# =============================================================================
# 3. Data preparation and cache
# =============================================================================
def speed_magnitude(u, v):
    """Return vector speed magnitude."""
    return np.sqrt(u * u + v * v)


def cache_is_compatible(cache):
    """Check whether a saved pointwise cache matches current thresholds."""
    required = {
        "cache_version",
        "slab_min_speed",
        "gt_min_speed",
        "x",
        "gt",
        "h_filtered",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and np.isclose(float(cache["slab_min_speed"]), SLAB_MIN_SPEED)
        and np.isclose(float(cache["gt_min_speed"]), GT_MIN_SPEED)
    )


def compute_pointwise_cache():
    """Extract finite pointwise slab speed, reference speed, and MLD values."""
    if not RAW_NPZ.exists():
        raise FileNotFoundError(
            f"Source NPZ file not found:\n  {RAW_NPZ}\n"
            "Place drifter_niw_300h_new.npz in the project data folder."
        )
    if not IDX_GOOD_NPY.exists():
        raise FileNotFoundError(
            f"Good-segment index file not found:\n  {IDX_GOOD_NPY}\n"
            "Place idx_good.npy in the project data folder."
        )

    idx_good = np.load(IDX_GOOD_NPY).astype(np.int64)

    with np.load(RAW_NPZ, mmap_mode="r") as source:
        required = {"X", "Y"}
        if not required.issubset(source.files):
            raise KeyError(
                f"{RAW_NPZ.name} must contain keys {sorted(required)}. "
                f"Available keys: {sorted(source.files)}"
            )

        x_good = source["X"][idx_good]
        y_good = source["Y"][idx_good]

        u_slab = x_good[:, :, 8]
        v_slab = x_good[:, :, 9]
        u_gt = y_good[:, :, 0]
        v_gt = y_good[:, :, 1]
        mld = x_good[:, :, 4]

        slab_speed = speed_magnitude(u_slab, v_slab).reshape(-1)
        gt_speed = speed_magnitude(u_gt, v_gt).reshape(-1)
        mld = mld.reshape(-1)

    valid = np.isfinite(slab_speed) & np.isfinite(gt_speed) & np.isfinite(mld)
    slab_speed = slab_speed[valid]
    gt_speed = gt_speed[valid]
    mld = mld[valid]

    speed_mask = (slab_speed >= SLAB_MIN_SPEED) & (gt_speed >= GT_MIN_SPEED)
    x = slab_speed[speed_mask]
    gt = gt_speed[speed_mask]
    h_filtered = mld[speed_mask]

    np.savez_compressed(
        CACHE_FILE,
        cache_version=np.array(CACHE_VERSION, dtype=np.int16),
        slab_min_speed=np.array(SLAB_MIN_SPEED, dtype=float),
        gt_min_speed=np.array(GT_MIN_SPEED, dtype=float),
        x=x,
        gt=gt,
        h_filtered=h_filtered,
    )


def load_or_build_pointwise_data():
    """Load a compatible cache or rebuild it from the source arrays."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return (
                    np.asarray(cache["x"], dtype=float),
                    np.asarray(cache["gt"], dtype=float),
                    np.asarray(cache["h_filtered"], dtype=float),
                )

    compute_pointwise_cache()
    with np.load(CACHE_FILE, allow_pickle=False) as cache:
        return (
            np.asarray(cache["x"], dtype=float),
            np.asarray(cache["gt"], dtype=float),
            np.asarray(cache["h_filtered"], dtype=float),
        )


x, gt, h_filtered = load_or_build_pointwise_data()


# =============================================================================
# 4. Diagnostic metric
# =============================================================================
def calculate_metric(metric_type, slab_speed, reference_speed):
    """Calculate the selected pointwise diagnostic metric."""
    if metric_type == "RATIO":
        values = reference_speed / (slab_speed + EPS)
        ylabel = "Speed ratio (|GT| / |Slab|)"
    elif metric_type == "SMOOTHED_RATIO":
        values = reference_speed / (slab_speed + SMOOTH_C)
        ylabel = rf"Smoothed ratio (|GT| / (|Slab| + {SMOOTH_C:g}))"
    elif metric_type == "SMAPE":
        values = np.abs(reference_speed - slab_speed) / (
            np.abs(reference_speed) + np.abs(slab_speed) + EPS
        )
        ylabel = "SMAPE"
    elif metric_type == "ABS_ERROR":
        values = np.abs(reference_speed - slab_speed)
        ylabel = r"Absolute error (m s$^{-1}$)"
    else:
        raise ValueError(f"Unsupported METRIC_TYPE: {metric_type}")

    return values, ylabel


y, y_label = calculate_metric(METRIC_TYPE, x, gt)


# =============================================================================
# 5. Plotting helpers
# =============================================================================
def add_binned_curve(ax, x_data, y_data, nbins=NBINS, xlog=False, stat="median"):
    """Overlay the binned median or mean diagnostic curve."""
    if xlog:
        edges = np.logspace(
            np.log10(max(np.nanmin(x_data), 1e-3)),
            np.log10(np.nanmax(x_data)),
            nbins + 1,
        )
    else:
        edges = np.linspace(np.nanmin(x_data), np.nanmax(x_data), nbins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        selected = (x_data >= edges[i]) & (x_data < edges[i + 1])
        if np.count_nonzero(selected) < 30:
            continue
        if stat == "mean":
            binned[i] = np.nanmean(y_data[selected])
        else:
            binned[i] = np.nanmedian(y_data[selected])

    valid = np.isfinite(binned)
    white_outline = [pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()]
    ax.plot(
        centers[valid],
        binned[valid],
        color="#D62728",
        linewidth=1.35,
        path_effects=white_outline,
        label="Binned median" if stat == "median" else "Binned mean",
        zorder=5,
    )


# =============================================================================
# 6. Supplementary Figure S8
# =============================================================================
fig = plt.figure(figsize=(5.8, 4.35), facecolor="white")
ax = fig.add_axes([0.12, 0.14, 0.69, 0.80])
cax = fig.add_axes([0.845, 0.22, 0.025, 0.64])

# Restrict the visible hexbin range using the same robust percentiles as the
# original diagnostic plot.
xlo, xhi = np.nanpercentile(x, (0.0, 99.9))
ylo, yhi = np.nanpercentile(y, (0.0, 99.0))
extent = (max(0.0, xlo), xhi, max(0.0, ylo), yhi)

hexbin = ax.hexbin(
    x,
    y,
    C=h_filtered,
    reduce_C_function=np.nanmedian,
    gridsize=GRIDSIZE,
    mincnt=MINCNT,
    extent=extent,
    cmap="hsv",
    linewidths=0,
    alpha=0.85,
    rasterized=True,
    zorder=2,
)

cbar = fig.colorbar(hexbin, cax=cax, orientation="vertical", extend="max")
cbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
cbar.outline.set_linewidth(0.8)
cbar.ax.text(
    0.5,
    1.055,
    "MLD (m)",
    transform=cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.0,
)

if METRIC_TYPE in {"RATIO", "SMOOTHED_RATIO"}:
    ax.axhline(
        1.0,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="Ideal ratio",
        zorder=4,
    )
else:
    ax.axhline(
        0.0,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="Zero error",
        zorder=4,
    )

add_binned_curve(ax, x, y, nbins=NBINS, xlog=X_LOG, stat="median")

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.7,
    color="#B0B0B0",
    alpha=0.35,
    zorder=0,
)

if X_LOG:
    ax.set_xscale("log")
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        color="#C7C7C7",
        alpha=0.30,
        zorder=0,
    )

ax.set_xlim(XLIM)
if YLIM is not None:
    ax.set_ylim(YLIM)
elif METRIC_TYPE == "SMAPE":
    ax.set_ylim(0.0, 1.05)
else:
    ax.set_ylim(extent[2], extent[3])

ax.set_xlabel(r"Pointwise slab speed (m s$^{-1}$)")
ax.set_ylabel(y_label)
ax.tick_params(axis="both", width=0.8, length=3)
ax.legend(
    loc="upper right",
    frameon=False,
    handlelength=1.8,
    borderaxespad=0.25,
    labelspacing=0.25,
    handletextpad=0.45,
    fontsize=7.0,
)

# Save as a 400-dpi RGB PNG.
fig.savefig(
    FIGURE_FILE,
    dpi=400,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)

try:
    from PIL import Image

    with Image.open(FIGURE_FILE) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(FIGURE_FILE, dpi=(400, 400))
except ImportError as exc:
    raise ImportError(
        "Pillow is required to ensure RGB PNG output. Install it with "
        "'pip install pillow' or 'conda install pillow'."
    ) from exc

plt.show()
