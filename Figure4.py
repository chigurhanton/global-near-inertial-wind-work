# -*- coding: utf-8 -*-
"""
Figure4_abc_clean_final.py

Panels (a), (b): keep the original absolute linear-trend maps unchanged.
Panel (c): zonal-mean trends with distinct, explicit definitions:
    V_NI or V_total:  100 * (ordinary OLS slope per decade) / (1979-2025 mean)
                       unit: % decade^-1 (relative to the long-term mean).
    F_NI:             ordinary OLS slope of the fraction * 100 per decade
                       unit: percentage points decade^-1.

Unlike the old Figure4_abc_dual.py, this script NEVER uses cached logarithmic
slope variables (G_ni_log_pct_dec / G_total_log_pct_dec).

This version REUSES spatial OLS trends already calculated:
  * Fig. 4a, 4b: data/Figure4_plot_data_final.nc
  * Fig. 3a: cache/Fig3_windstress_plot_data.npz (slope_ni_dec)
And reads annual V_NI, V_total from the ORIGINAL stress NetCDF solely to
compute the 1979-2025 means and complete-record mask (no repeated regression,
FFT, or slab simulation).
Coordinate alignment and a cross-check between Fig. 3 and Fig. 4 total-variance
trend grids prevent silent mismatches. Fig. 3a and Fig. 4a, b stay unchanged.

CLEAN VERSION: Panel c shows ONLY the three trend curves. No HAC, CI ribbons,
error bars or extra significance/diagnostic figure are calculated or drawn.
The original p<0.05 map hatching in panels a and b remains unchanged.

"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 1. FILES
# =============================================================================
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
PLOTS_DIR = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = DATA_DIR / "Figure4_plot_data_final.nc"
FIG3_CACHE_FILE = PROJECT_DIR / "cache" / "Fig3_windstress_plot_data.npz"
STRESS_FILE = DATA_DIR / "global_wind_stress_ni_total_variance_1979_2025.nc"
NI_VARIABLE = "resonant_ni_wind_stress_variance"
TOTAL_VARIABLE = "total_wind_stress_variance"
FIGURE_PNG = PLOTS_DIR / "Figure4.png"
FIGURE_PDF = PLOTS_DIR / "Figure4.pdf"


# =============================================================================
# 2. FINAL PHYSICAL SIZE
# =============================================================================
FIG_WIDTH_IN = 5.4
FIG_HEIGHT_IN = 7.8


# =============================================================================
# 3. LAYOUT CONTROLS
# =============================================================================
LAYOUT = {
    "LEFT_MARGIN": 0.045,
    "RIGHT_MARGIN": 0.035,
    "BOTTOM_ROW_Y": 0.235,
    "MAP_SLOT_HEIGHT": 0.260,
    "COLUMN_GAP": 0.060,
    "ROW_GAP": 0.095,

    "CB_WIDTH_FRACTION": 0.58,
    "CB_HEIGHT": 0.013,
    "CB_CLEAR_GAP_TOP": 0.012,
    "CB_CLEAR_GAP_BOTTOM": 0.030,

    "LABEL_DX": 0.008,
    "LABEL_DY": 0.007,
    "LABEL_FONTSIZE": 8.0,
}


# =============================================================================
# 4. DISPLAY / STYLE CONTROLS
# =============================================================================
P_THRESHOLD = 0.05
MAP_PERCENTILE = 98.0
CONTINUOUS_HATCH_COLOR = "#000000"

# Total-variance trend is scaled so the colorbar ticks are clean numbers.
# The displayed unit therefore carries the 10^-3 factor.
TOTAL_SCALE = 1.0e3

mpl.rcParams.update({
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

CMAP_TOTAL = mpl.colormaps["RdBu_r"]
CMAP_FRACTION = mpl.colormaps["BrBG_r"]


# =============================================================================
# 5. SMALL HELPERS
# =============================================================================
def robust_symmetric_limit(da, percentile=MAP_PERCENTILE):
    values = np.asarray(da.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    value = float(np.nanpercentile(np.abs(values), percentile))
    if not np.isfinite(value) or value <= 0.0:
        value = float(np.nanmax(np.abs(values)))
    return value if np.isfinite(value) and value > 0.0 else 1.0


def nice_ceiling(value):
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    scaled = value / scale
    candidates = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    upper = candidates[candidates >= scaled]
    return float((upper[0] if upper.size else 10.0) * scale)


def add_left_latitude_labels(ax, latitudes=(-60, -30, 0, 30, 60), fontsize=6.8):
    latitudes = np.asarray(latitudes, dtype=float)
    points = ax.projection.transform_points(
        ccrs.PlateCarree(), np.zeros_like(latitudes), latitudes
    )
    x_min, x_max = ax.get_xlim()
    x_offset = 0.012 * (x_max - x_min)

    for lat_value, (x_proj, y_proj, _) in zip(latitudes, points):
        if lat_value < 0:
            label = f"{int(abs(lat_value))}°S"
        elif lat_value > 0:
            label = f"{int(lat_value)}°N"
        else:
            label = "0°"

        ax.text(
            x_proj - x_offset,
            y_proj,
            label,
            transform=ax.transData,
            ha="right",
            va="center",
            fontsize=fontsize,
            color="#333333",
            clip_on=False,
            zorder=10,
        )


def add_map_gridlines(ax, show_bottom_labels=False):
    grid = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.7,
        color="#666666",
        alpha=0.45,
        linestyle="--",
        zorder=7,
    )
    grid.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    grid.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    if show_bottom_labels:
        labels = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.0
        )
        labels.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        labels.ylocator = mticker.FixedLocator([])
        labels.top_labels = False
        labels.bottom_labels = True
        labels.left_labels = False
        labels.right_labels = False
        labels.xlabel_style = {"size": 6.8, "color": "#333333"}
        if hasattr(labels, "geo_labels"):
            labels.geo_labels = False
        try:
            labels.x_inline = False
        except AttributeError:
            pass


def set_hatch_edgecolor(contour_set, color):
    try:
        contour_set.set_edgecolor(color)
    except Exception:
        pass
    try:
        contour_set.set_linewidth(0.0)
    except Exception:
        pass
    try:
        contour_set.set_linewidths([0.0])
    except Exception:
        pass

    try:
        collections = contour_set.collections
    except Exception:
        collections = []

    for collection in collections:
        try:
            collection.set_edgecolor(color)
        except Exception:
            pass
        try:
            collection.set_linewidth(0.0)
        except Exception:
            pass
        try:
            collection.set_linewidths([0.0])
        except Exception:
            pass


def add_single_color_hatching(ax, da, pvalue, threshold=P_THRESHOLD,
                               color=CONTINUOUS_HATCH_COLOR):
    data = np.asarray(da.values, dtype=float)
    p = np.asarray(pvalue.values, dtype=float)
    sig = np.where(
        np.isfinite(data) & np.isfinite(p) & (p < threshold),
        1.0,
        np.nan,
    )
    cyclic_sig, cyclic_lon = add_cyclic_point(sig, coord=da["lon"].values)
    if np.any(np.isfinite(cyclic_sig)):
        cs = ax.contourf(
            cyclic_lon,
            da["lat"].values,
            cyclic_sig,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            antialiased=False,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        set_hatch_edgecolor(cs, color)


def plot_continuous_map(ax, da, pvalue, cmap, vmin, vmax):
    data = np.asarray(da.values, dtype=float)
    cyclic_data, cyclic_lon = add_cyclic_point(data, coord=da["lon"].values)

    image = ax.pcolormesh(
        cyclic_lon,
        da["lat"].values,
        cyclic_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
        zorder=1,
    )

    add_single_color_hatching(ax, da, pvalue)

    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=4)
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=8)
    ax.set_global()
    return image


def visible_map_bbox(ax, fig, renderer):
    return ax.patch.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )


def place_horizontal_cbar(
    fig,
    image,
    cax_pos,
    ticks,
    unit_text=None,
):
    cax = fig.add_axes(cax_pos)
    cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=7.2, width=0.8, length=3, pad=2)
    cb.outline.set_linewidth(0.8)
    if unit_text:
        cb.ax.text(
            1.03,
            0.5,
            unit_text,
            transform=cb.ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
        )
    return cb


def add_panel_letter(fig, x, y, label):
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=LAYOUT["LABEL_FONTSIZE"],
        fontweight="bold",
    )


# =============================================================================
# 6. REUSE EXISTING SPATIAL TRENDS; ONLY CALCULATE ANNUAL MEANS
# =============================================================================
YEAR_START, YEAR_END = 1979, 2025
COORD_TOLERANCE = 1.0e-3  # degrees; NO spatial interpolation


def normalize_lon_grid(field):
    """Convert a lat-lon field to sorted [0, 360) longitudes."""
    if "latitude" in field.dims:
        field = field.rename({"latitude": "lat"})
    if "longitude" in field.dims:
        field = field.rename({"longitude": "lon"})
    field = field.assign_coords(lon=(field.lon % 360.0)).sortby("lon")
    field = field.sortby("lat")
    if len(np.unique(np.round(field.lon.values, 6))) != field.sizes["lon"]:
        raise ValueError("Duplicated longitude coordinates after 0-360 normalization.")
    return field


def align_exact_grid(field, reference, label):
    """Check that grids match before assigning identical float coordinates.

    This must not silently interpolate/smooth the Fig. 3 trends.
    """
    field = normalize_lon_grid(field)
    reference = normalize_lon_grid(reference)
    if field.sizes["lat"] != reference.sizes["lat"] or field.sizes["lon"] != reference.sizes["lon"]:
        raise ValueError(
            f"{label}: grid shapes differ. Fig3={field.shape[-2:]}, "
            f"Fig4={reference.shape[-2:]}. Need identical spatial grids."
        )
    if not np.allclose(field.lat.values, reference.lat.values, rtol=0.0, atol=COORD_TOLERANCE):
        raise ValueError(f"{label}: latitude coordinates of Fig. 3 and Fig. 4 differ.")
    if not np.allclose(field.lon.values, reference.lon.values, rtol=0.0, atol=COORD_TOLERANCE):
        raise ValueError(f"{label}: longitude coordinates of Fig. 3 and Fig. 4 differ.")
    field = field.assign_coords(lat=reference.lat, lon=reference.lon)
    return field.transpose("lat", "lon")


def load_fig3_raw_trends(reference):
    if not FIG3_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Missing Fig. 3 plot cache: {FIG3_CACHE_FILE}\n"
            "Run Figure3_windstress_newcolorbar.py (or recover its plot NPZ) first. "
            "It must contain slope_ni_dec and slope_total_dec."
        )
    with np.load(FIG3_CACHE_FILE, allow_pickle=False) as data:
        required = {"lat", "lon", "slope_ni_dec", "slope_total_dec"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing {sorted(missing)} in {FIG3_CACHE_FILE}.")
        for name, expected_year in (("start_year", YEAR_START), ("end_year", YEAR_END)):
            if name in data and int(data[name]) != expected_year:
                raise ValueError(f"Fig. 3 cache has inconsistent {name}={int(data[name])}.")
        coords = {"lat": data["lat"].astype(float), "lon": data["lon"].astype(float)}
        ni = xr.DataArray(data["slope_ni_dec"].astype(float), coords=coords, dims=("lat", "lon"))
        total = xr.DataArray(data["slope_total_dec"].astype(float), coords=coords, dims=("lat", "lon"))
    return (align_exact_grid(ni, reference, "Fig. 3 near-inertial trend"),
            align_exact_grid(total, reference, "Fig. 3 total-variance trend"))


def annual_climatological_mean_and_complete_mask(field, label):
    """Compute 1979–2025 average and full-record valid mask; no regression.

    Only cells with strictly positive, finite annual data in all 47 years are
    used in panel c, consistent with the last HAC version's central curves.
    """
    field = normalize_lon_grid(field)
    if "time" not in field.dims or not np.issubdtype(field.time.dtype, np.datetime64):
        raise ValueError(f"{label}: expected datetime64 time axis.")
    years = np.asarray(field.time.dt.year.values, dtype=int)
    keep = (years >= YEAR_START) & (years <= YEAR_END)
    field = field.isel(time=np.flatnonzero(keep))
    years = years[keep]
    expected = np.arange(YEAR_START, YEAR_END + 1)
    if not np.array_equal(np.unique(years), expected):
        raise ValueError(f"{label}: incomplete {YEAR_START}-{YEAR_END} records.")
    # Annual data are normally one record per year; if finer than annual,
    # compute calendar-year means exactly as the Fig. 3 workflow does.
    if len(years) == len(expected):
        annual = field.rename(time="year").assign_coords(year=years)
    else:
        annual = field.groupby("time.year").mean("time", skipna=True)
    annual = annual.sortby("year")
    climatology = annual.mean("year", skipna=True).load()
    complete = (np.isfinite(annual) & (annual > 0)).all("year").load()
    return climatology, complete


def load_existing_trends_and_means():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Missing Fig. 4 cache: {CACHE_FILE}")
    if not STRESS_FILE.exists():
        raise FileNotFoundError(f"Missing annual wind-stress data: {STRESS_FILE}")

    with xr.open_dataset(CACHE_FILE) as ds:
        keys = ["total_linear_trend_per_decade", "total_linear_pvalue",
                "fraction_linear_trend_pp_dec", "fraction_linear_pvalue"]
        missing = [k for k in keys if k not in ds]
        if missing:
            raise KeyError(f"Missing Fig. 4 map variables: {missing}")
        total_trend_raw = normalize_lon_grid(ds[keys[0]].load())
        total_pvalue = align_exact_grid(ds[keys[1]].load(), total_trend_raw, keys[1])
        fraction_trend = align_exact_grid(ds[keys[2]].load(), total_trend_raw, keys[2])
        fraction_pvalue = align_exact_grid(ds[keys[3]].load(), total_trend_raw, keys[3])

    # Reuse both spatial trends from Fig. 3; total trend is diagnostic only.
    ni_trend_raw, fig3_total_trend = load_fig3_raw_trends(total_trend_raw)
    comparable = np.isfinite(fig3_total_trend.values) & np.isfinite(total_trend_raw.values)
    if comparable.any():
        delta = np.abs(fig3_total_trend.values[comparable] - total_trend_raw.values[comparable])
        reference_scale = np.nanpercentile(np.abs(total_trend_raw.values[comparable]), 95)
        # Some local trends approach zero; compare aggregate scale, not pointwise ratio.
        discrepancy = float(np.nanpercentile(delta, 95) / max(reference_scale, 1.0e-20))
        print(f"Fig3 vs Fig4 total-variance trend: 95th percentile difference / trend scale = {discrepancy:.3%}")
        if discrepancy > 0.02:
            raise ValueError(
                "Fig. 3 and Fig. 4 total-variance trend fields disagree by >2% "
                "in a robust check; inspect years, grid, valid-data mask before "
                "combining the panels."
            )

    with xr.open_dataset(STRESS_FILE) as ds:
        for name in (NI_VARIABLE, TOTAL_VARIABLE):
            if name not in ds:
                raise KeyError(f"Missing annual field {name!r} in {STRESS_FILE.name}.")
        ni_mean, complete_ni = annual_climatological_mean_and_complete_mask(ds[NI_VARIABLE], NI_VARIABLE)
        total_mean, complete_total = annual_climatological_mean_and_complete_mask(ds[TOTAL_VARIABLE], TOTAL_VARIABLE)

    ni_mean = align_exact_grid(ni_mean, total_trend_raw, "Mean near-inertial variance")
    total_mean = align_exact_grid(total_mean, total_trend_raw, "Mean total variance")
    complete_ni = align_exact_grid(complete_ni, total_trend_raw, "Near-inertial complete-year mask")
    complete_total = align_exact_grid(complete_total, total_trend_raw, "Total complete-year mask")
    ni_mean = ni_mean.where(np.isfinite(ni_trend_raw) & (ni_mean > 0.0))
    total_mean = total_mean.where(np.isfinite(total_trend_raw) & (total_mean > 0.0))
    print("Reusing existing spatial trends; computed 47-year means and fixed ocean mask only.")
    return (total_trend_raw, total_pvalue, fraction_trend, fraction_pvalue,
            ni_trend_raw, ni_mean, total_mean, complete_ni, complete_total)


(total_trend_raw, total_pvalue, fraction_trend, fraction_pvalue,
 ni_trend_raw, ni_mean, total_mean, complete_ni, complete_total) = load_existing_trends_and_means()
total_trend = total_trend_raw * TOTAL_SCALE  # scaling ONLY for map-a colorbar

# Relative trends: original spatial OLS slope per decade / long-term mean.
relative_ni = (100.0 * ni_trend_raw / ni_mean).transpose("lat", "lon")
relative_total = (100.0 * total_trend_raw / total_mean).transpose("lat", "lon")
relative_ni = relative_ni.where(np.isfinite(relative_ni))
relative_total = relative_total.where(np.isfinite(relative_total))

lat = fraction_trend.lat.values.astype(float)
common = (np.isfinite(relative_ni.values) & np.isfinite(relative_total.values)
          & np.isfinite(fraction_trend.values)
          & complete_ni.values.astype(bool) & complete_total.values.astype(bool))
print(f"Panel c uses {int(common.sum())} complete-record ocean cells.")


def zonal_profile(field):
    """Arithmetic longitude mean of grid-cell trends on a common valid mask."""
    arr = np.asarray(field, dtype=float)
    counts = common.sum(axis=1)
    totals = np.sum(np.where(common, arr, 0.0), axis=1)
    return np.divide(totals, counts, out=np.full(arr.shape[0], np.nan), where=counts > 0)


prof_ni = zonal_profile(relative_ni.values)
prof_total = zonal_profile(relative_total.values)
prof_frac_abs = zonal_profile(fraction_trend.values)

total_limit = nice_ceiling(robust_symmetric_limit(total_trend))
fraction_limit = nice_ceiling(robust_symmetric_limit(fraction_trend))


# =============================================================================
# 7. MAIN POSITIONS (follow Figure3_windstress_v2.png layout)
# =============================================================================
pos_a = [0.08, 0.705, 0.84, 0.235]
pos_b = [0.08, 0.395, 0.84, 0.235]
pos_c = [0.14, 0.09, 0.72, 0.20]

cax_pos_a = [0.29, 0.675, 0.42, 0.012]
cax_pos_b = [0.29, 0.34, 0.42, 0.012]


# =============================================================================
# 8. DRAW TWO MAP PANELS
# =============================================================================
fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="white")
projection = ccrs.Robinson(central_longitude=180)

ax_a = fig.add_axes(pos_a, projection=projection)
ax_b = fig.add_axes(pos_b, projection=projection)

# (a) Ordinary linear trend in total wind-stress variance.
im_a = plot_continuous_map(
    ax_a,
    total_trend,
    total_pvalue,
    CMAP_TOTAL,
    -total_limit,
    total_limit,
)
add_map_gridlines(ax_a, show_bottom_labels=False)
add_left_latitude_labels(ax_a)

# (b) Ordinary linear trend in F_NI.
im_b = plot_continuous_map(
    ax_b,
    fraction_trend,
    fraction_pvalue,
    CMAP_FRACTION,
    -fraction_limit,
    fraction_limit,
)
add_map_gridlines(ax_b, show_bottom_labels=True)
add_left_latitude_labels(ax_b)


cb_a = place_horizontal_cbar(
    fig,
    im_a,
    cax_pos_a,
    np.linspace(-total_limit, total_limit, 5),
    unit_text=r"(10$^{-3}$ N$^2$ m$^{-4}$ decade$^{-1}$)",
)

cb_b = place_horizontal_cbar(
    fig,
    im_b,
    cax_pos_b,
    np.linspace(-fraction_limit, fraction_limit, 5),
    unit_text=r"(percentage points decade$^{-1}$)",
)

# =============================================================================
# 9. PANEL C: DUAL-AXIS ZONAL-MEAN TRENDS
# =============================================================================
ax_c = fig.add_axes(pos_c)
ax_c2 = ax_c.twinx()

# Left axis: ordinary OLS trend divided by long-term arithmetic mean.
# Both are % per decade, NOT log-linear slopes.
ax_c.plot(lat, prof_ni, color="#111111", linewidth=1.3, zorder=5,
          label=r"Near-inertial wind-stress variance ($V_{NI}$)")
ax_c.plot(lat, prof_total, color="#111111", linewidth=1.3, linestyle="--"
          , zorder=4,
          label=r"Total wind-stress variance ($V_{total}$)")

# Right axis: ordinary linear trend in fraction, in percentage points/decade.
ax_c2.plot(lat, prof_frac_abs, color="#b2182b", linewidth=1.3, zorder=5,
           label=r"Near-inertial spectral fraction ($F_{NI}$)")

ax_c.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
ax_c.set_xlim(-60, 60)
ax_c.set_xticks([-60, -30, 0, 30, 60])
ax_c.set_xticklabels(["60\u00b0S", "30\u00b0S", "0\u00b0", "30\u00b0N", "60\u00b0N"])
ax_c.set_ylabel("Relative trend (% decade$^{-1}$)", fontsize=7.6,
                color="#111111", labelpad=4)
ax_c2.set_ylabel("Spectral fraction trend\n(percentage points decade$^{-1}$)",
                 fontsize=7.2, color="#b2182b", labelpad=4)
ax_c.tick_params(axis="y", colors="#111111")
ax_c2.tick_params(axis="y", colors="#b2182b")
ax_c2.spines["right"].set_edgecolor("#b2182b")

# Match the zero line on both y axes, preserving all three central curves.
def align_zeros_from_centres(left, right):
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return
    a_min, a_max = min(float(a.min()), 0.), max(float(a.max()), 0.)
    b_min, b_max = min(float(b.min()), 0.), max(float(b.max()), 0.)
    af = -a_min / max(a_max-a_min, 1.e-15)
    bf = -b_min / max(b_max-b_min, 1.e-15)
    frac = float(np.clip((af + bf)/2, 0.13, 0.87))
    for ax, vmin, vmax in ((ax_c, a_min, a_max), (ax_c2, b_min, b_max)):
        span = max(-vmin/frac, vmax/(1-frac), 1.e-12)*1.09
        ax.set_ylim(-frac*span, (1-frac)*span)

align_zeros_from_centres(np.r_[prof_ni, prof_total], prof_frac_abs)

ax_c.grid(color="#E4E4E4", linewidth=0.6, linestyle="-", zorder=0)

lines1, labels1 = ax_c.get_legend_handles_labels()
lines2, labels2 = ax_c2.get_legend_handles_labels()
ax_c.legend(
    lines1 + lines2,
    labels1 + labels2,
    frameon=False,
    loc="upper left",
    fontsize=6.0,
    handlelength=1.9,
    labelspacing=0.3,
    borderaxespad=0.25,
    bbox_to_anchor=(0.015, 1.0),
)


# =============================================================================
# 10. PANEL LETTERS
# =============================================================================
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bbox_a = visible_map_bbox(ax_a, fig, renderer)
bbox_b = visible_map_bbox(ax_b, fig, renderer)
bbox_c = ax_c.get_position()

add_panel_letter(
    fig,
    bbox_a.x0 - LAYOUT["LABEL_DX"],
    bbox_a.y1 + LAYOUT["LABEL_DY"],
    "a",
)
add_panel_letter(
    fig,
    bbox_b.x0 - LAYOUT["LABEL_DX"],
    bbox_b.y1 + LAYOUT["LABEL_DY"],
    "b",
)
add_panel_letter(
    fig,
    bbox_c.x0 - LAYOUT["LABEL_DX"],
    bbox_c.y1 + LAYOUT["LABEL_DY"],
    "c",
)


# =============================================================================
# 11. SAVE AT EXACT PHYSICAL SIZE
# =============================================================================
fig.savefig(
    FIGURE_PNG,
    dpi=400,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)
fig.savefig(
    FIGURE_PDF,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)

try:
    from PIL import Image

    with Image.open(FIGURE_PNG) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(FIGURE_PNG, dpi=(400, 400))
except ImportError:
    pass

plt.show()


# =============================================================================
# 12. CONSOLE DIAGNOSTICS
# =============================================================================
print("\nSaved:")
print(f"  {FIGURE_PNG}")
print(f"  {FIGURE_PDF}")
print("\nPanel definitions:")
print("  a: ordinary linear trend of V_total (per decade)")
print("  b: ordinary linear trend of F_NI (percentage points per decade)")
print("  c: left = ordinary OLS trend/1979-2025 mean * 100 (%/decade)")
print("     right = ordinary trend of F_NI (percentage points/decade)")
print("  c: clean curves only; no HAC calculations, CI bands, error bars or reference plots")
print("\nPhysical canvas size:")
print(f"  {FIG_WIDTH_IN * 25.4:.1f} mm x {FIG_HEIGHT_IN * 25.4:.1f} mm")
print(f"  PNG @400 dpi ~ {round(FIG_WIDTH_IN * 400)} x {round(FIG_HEIGHT_IN * 400)} px")
