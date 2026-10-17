# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S8.

Single-panel start–end sensitivity heatmap for the North Pacific
background-adjusted global near-inertial wind-work series.

The required annual series is read from the latest Figure 1 plotting cache.
Designed for execution as a single cell in a VS Code Jupyter notebook.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import statsmodels.api as sm


# =============================================================================
# 0. PROJECT PATHS
# =============================================================================
PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "cache"
PLOTS_DIR = PROJECT_DIR / "plots"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FIG1_CACHE = CACHE_DIR / "Fig1_plot_data.npz"
S6_CACHE = CACHE_DIR / "FigS8_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS8.png"

CACHE_VERSION = 1
RECOMPUTE_CACHE = False


# =============================================================================
# 1. SETTINGS
# =============================================================================
MIN_SPAN = 25
START_YEARS = np.arange(1979, 2002)   # 1979 ... 2001
END_YEARS = np.arange(2003, 2026)     # 2003 ... 2025
HAC_LAG = 3
VMAX = 10.0
OUTLINE_TARGET_START_YEARS = {1997, 1998}
OUTLINE_COLOR = "#E41A1C"
OUTLINE_LINEWIDTH = 1.0


# =============================================================================
# 2. STYLE
# =============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "axes.linewidth": 0.8,
})


# =============================================================================
# 3. UTILITIES
# =============================================================================
def calc_trend(series, start_year, end_year):
    """Return linear trend (GW decade^-1) and HAC(3) P value."""
    mask = (series["year"] >= start_year) & (series["year"] <= end_year)
    years = series["year"][mask].astype(float)
    values = series["value"][mask].astype(float)
    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    values = values[valid]

    if len(values) < 4:
        return np.nan, np.nan

    X = sm.add_constant(years - years.mean())
    fit = sm.OLS(values, X).fit()
    robust_fit = fit.get_robustcov_results(
        cov_type="HAC",
        maxlags=min(HAC_LAG, len(values) - 1),
    )

    trend_gw_per_dec = float(fit.params[1] * 10000.0)  # TW yr^-1 -> GW decade^-1
    p_value = float(robust_fit.pvalues[1])
    return trend_gw_per_dec, p_value


def find_connected_components(mask):
    """Return 4-neighbour connected components of a 2-D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    nrow, ncol = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for i in range(nrow):
        for j in range(ncol):
            if (not mask[i, j]) or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= rr < nrow
                        and 0 <= cc < ncol
                        and mask[rr, cc]
                        and not visited[rr, cc]
                    ):
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            components.append(cells)
    return components


# =============================================================================
# 4. LOAD FIGURE 1 CACHE AND BUILD S6 CACHE
# =============================================================================
def cache_is_compatible(cache):
    required = {
        "cache_version",
        "start_years",
        "end_years",
        "min_span",
        "hac_lag",
        "vmax",
        "trend_matrix",
        "p_matrix",
        "sig_pos_mask",
        "nonsig_mask",
        "sig_pos_pct",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["min_span"]) == MIN_SPAN
        and int(cache["hac_lag"]) == HAC_LAG
        and float(cache["vmax"]) == VMAX
    )


def compute_s6_cache():
    if not FIG1_CACHE.exists():
        raise FileNotFoundError(
            "Figure S6 requires the latest Figure 1 plotting cache.\n\n"
            f"Expected file:\n  {FIG1_CACHE}\n\n"
            "Please run the latest Figure 1 code first."
        )

    with np.load(FIG1_CACHE, allow_pickle=False) as cache:
        required = {"years", "global_background_adjusted_tw"}
        if not required.issubset(cache.files):
            raise KeyError(
                "The Figure 1 cache does not contain the required annual series.\n"
                f"Required keys: {sorted(required)}\n"
                f"Available keys: {sorted(cache.files)}"
            )
        years = np.asarray(cache["years"], dtype=int)
        values = np.asarray(cache["global_background_adjusted_tw"], dtype=float)

    order = np.argsort(years)
    years = years[order]
    values = values[order]
    mask = (years >= 1979) & (years <= 2025)
    series = {
        "year": years[mask],
        "value": values[mask],
    }

    shape = (len(END_YEARS), len(START_YEARS))
    trend_matrix = np.full(shape, np.nan, dtype=float)
    p_matrix = np.full(shape, np.nan, dtype=float)

    for i, end_year in enumerate(END_YEARS):
        for j, start_year in enumerate(START_YEARS):
            span = end_year - start_year + 1
            if span < MIN_SPAN:
                continue
            trend, p_value = calc_trend(series, start_year, end_year)
            trend_matrix[i, j] = trend
            p_matrix[i, j] = p_value

    valid_mask = np.isfinite(trend_matrix) & np.isfinite(p_matrix)
    sig_pos_mask = valid_mask & (trend_matrix > 0) & (p_matrix < 0.05)
    nonsig_mask = valid_mask & (p_matrix >= 0.05)

    n_valid = int(np.sum(valid_mask))
    n_sig_pos = int(np.sum(sig_pos_mask))
    sig_pos_pct = 100.0 * n_sig_pos / n_valid if n_valid > 0 else np.nan

    np.savez_compressed(
        S6_CACHE,
        cache_version=np.array(CACHE_VERSION, dtype=np.int16),
        start_years=START_YEARS.astype(np.int16),
        end_years=END_YEARS.astype(np.int16),
        min_span=np.array(MIN_SPAN, dtype=np.int16),
        hac_lag=np.array(HAC_LAG, dtype=np.int16),
        vmax=np.array(VMAX, dtype=float),
        trend_matrix=trend_matrix,
        p_matrix=p_matrix,
        sig_pos_mask=sig_pos_mask,
        nonsig_mask=nonsig_mask,
        sig_pos_pct=np.array(sig_pos_pct, dtype=float),
    )


def load_s6_cache():
    if S6_CACHE.exists() and not RECOMPUTE_CACHE:
        with np.load(S6_CACHE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {k: np.array(cache[k]) for k in cache.files}, "cache"

    compute_s6_cache()
    with np.load(S6_CACHE, allow_pickle=False) as cache:
        return {k: np.array(cache[k]) for k in cache.files}, "source"


cache_data, data_source = load_s6_cache()
trend_matrix = np.asarray(cache_data["trend_matrix"], dtype=float)
p_matrix = np.asarray(cache_data["p_matrix"], dtype=float)
sig_pos_mask = np.asarray(cache_data["sig_pos_mask"], dtype=bool)
nonsig_mask = np.asarray(cache_data["nonsig_mask"], dtype=bool)
sig_pos_pct = float(cache_data["sig_pos_pct"])


# =============================================================================
# 5. RED OUTLINE COMPONENTS
# =============================================================================
target_cols = {j for j, year in enumerate(START_YEARS) if int(year) in OUTLINE_TARGET_START_YEARS}
outlined_components = []
for component in find_connected_components(nonsig_mask):
    if any(j in target_cols for _, j in component):
        outlined_components.append(component)


def draw_component_outline(ax, cells, start_years, end_years, color=OUTLINE_COLOR, linewidth=OUTLINE_LINEWIDTH):
    """Draw only the outer edges of one connected cell patch."""
    cell_set = set(cells)
    for i, j in cells:
        x0 = float(start_years[j]) - 0.5
        x1 = float(start_years[j]) + 0.5
        y0 = float(end_years[i]) - 0.5
        y1 = float(end_years[i]) + 0.5

        if (i, j - 1) not in cell_set:
            ax.plot([x0, x0], [y0, y1], color=color, linewidth=linewidth, solid_capstyle="butt", zorder=7)
        if (i, j + 1) not in cell_set:
            ax.plot([x1, x1], [y0, y1], color=color, linewidth=linewidth, solid_capstyle="butt", zorder=7)
        if (i - 1, j) not in cell_set:
            ax.plot([x0, x1], [y0, y0], color=color, linewidth=linewidth, solid_capstyle="butt", zorder=7)
        if (i + 1, j) not in cell_set:
            ax.plot([x0, x1], [y1, y1], color=color, linewidth=linewidth, solid_capstyle="butt", zorder=7)


# =============================================================================
# 6. PLOT
# =============================================================================
start_edges = np.arange(START_YEARS[0] - 0.5, START_YEARS[-1] + 1.5, 1)
end_edges = np.arange(END_YEARS[0] - 0.5, END_YEARS[-1] + 1.5, 1)

cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("white")
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)

fig, ax = plt.subplots(figsize=(5.5, 5.05), facecolor="white")
fig.subplots_adjust(left=0.12, right=0.80, bottom=0.14, top=0.95)

mesh = ax.pcolormesh(
    start_edges,
    end_edges,
    np.ma.masked_invalid(trend_matrix),
    cmap=cmap,
    norm=norm,
    shading="flat",
    rasterized=True,
)

sig_i, sig_j = np.where(sig_pos_mask)
ax.scatter(
    START_YEARS[sig_j],
    END_YEARS[sig_i],
    s=14,
    color="black",
    linewidths=0,
    zorder=5,
)

for component in outlined_components:
    draw_component_outline(ax, component, START_YEARS, END_YEARS)

ax.axvline(1997, color="black", linestyle="--", linewidth=1.0, alpha=0.80, zorder=4)

ax.set_xlabel("Start year")
ax.set_ylabel("End year")
ax.set_xlim(1978.5, 2001.5)
ax.set_ylim(2002.5, 2025.5)
ax.set_xticks(np.arange(1979, 2002, 2))
ax.set_yticks(np.arange(2003, 2026, 2))
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="both", width=0.8, length=3)
for side in ["left", "right", "top", "bottom"]:
    ax.spines[side].set_linewidth(0.8)

# Significance legend — original framed style
legend_handle = ax.scatter(
    [],
    [],
    s=14,
    color="black",
    linewidths=0,
    label=r"Significantly positive ($P < 0.05$)",
)

sig_legend = ax.legend(
    handles=[legend_handle],
    loc="lower left",
    bbox_to_anchor=(0.455, 0.155),
    frameon=True,
    fancybox=True,
    fontsize=7.0,
    markerscale=1.0,
    handlelength=1.0,
    handletextpad=0.55,
    borderpad=0.48,
    labelspacing=0.35,
    framealpha=0.96,
)
sig_legend.get_frame().set_facecolor("#F6F8FA")
sig_legend.get_frame().set_edgecolor("#B8C1C9")
sig_legend.get_frame().set_linewidth(0.8)

# Statistics — original framed style
stats_text = (
    f"Minimum window: {MIN_SPAN} years\n"
    f"Significantly positive: {sig_pos_pct:.1f}%"
)
ax.text(
    0.475,
    0.055,
    stats_text,
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.0,
    linespacing=1.55,
    zorder=10,
    bbox=dict(
        boxstyle="round,pad=0.48",
        facecolor="#F6F8FA",
        edgecolor="#B8C1C9",
        linewidth=0.8,
        alpha=0.96,
    ),
)

# Vertical colorbar — original style, moved left to preserve the full label.
cax = fig.add_axes([0.825, 0.195, 0.022, 0.64])
cbar = fig.colorbar(mesh, cax=cax)
cbar.set_label(
    r"Trend (GW decade$^{-1}$)",
    fontsize=8.0,
    labelpad=7,
)
cbar.set_ticks(np.arange(-10, 10.1, 2.5))
cbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
cbar.outline.set_linewidth(0.8)

# Save as RGB PNG.
fig.savefig(FIGURE_FILE, dpi=400, facecolor="white", edgecolor="white", transparent=False)
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

print(f"Plot data: {data_source}")
print(f"Saved figure: {FIGURE_FILE}")
print(f"Plotting cache: {S6_CACHE}")
