# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S1 from source data or a plotting-data cache.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


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

CORR_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
SLAB_FILE = DATA_DIR / "slab_niw_power_input_1979_2025.nc"
CACHE_FILE = CACHE_DIR / "FigS1_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS1.png"

VAR_NAME = "power_input"
YEAR_START = 1979
YEAR_END = 2025
R_EARTH = 6_371_000.0
CACHE_VERSION = 1
RECOMPUTE_CACHE = False

# Difference direction used in panel a.
# "corrected_minus_slab": positive values indicate corrected > slab.
# "slab_minus_corrected": positive values indicate slab > corrected.
DIFF_MODE = "slab_minus_corrected"


# =============================================================================
# 2. Figure style
# =============================================================================
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


# =============================================================================
# 3. Utility functions
# =============================================================================
def standardize_latlon_names(da):
    """Rename latitude/longitude coordinates to lat/lon when needed."""
    rename_dict = {}
    if "latitude" in da.coords or "latitude" in da.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.coords or "longitude" in da.dims:
        rename_dict["longitude"] = "lon"
    return da.rename(rename_dict) if rename_dict else da


def open_power_input(file_path, var_name=VAR_NAME):
    """Read and load the requested wind-work field for the study period."""
    with xr.open_dataset(file_path, decode_times=True) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(f"Variable '{var_name}' was not found in {file_path.name}.")
        da = ds[var_name].sel(
            time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")
        )
        da = standardize_latlon_names(da)
        required = {"time", "lat", "lon"}
        missing = required - set(da.coords)
        if missing:
            raise KeyError(
                f"Missing coordinates {sorted(missing)} in {file_path.name}. "
                f"Available coordinates: {list(da.coords)}"
            )
        return da.load()


def compute_grid_area_2d(da):
    """Compute regular-grid spherical cell area on the DataArray grid."""
    lat = da["lat"]
    lon = da["lon"]

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Latitude and longitude coordinates must each contain at least two points.")

    dlat = float(np.abs(lat.values[1] - lat.values[0]))
    dlon = float(np.abs(lon.values[1] - lon.values[0]))

    area_1d = (
        R_EARTH**2
        * np.deg2rad(dlat)
        * np.deg2rad(dlon)
        * np.cos(np.deg2rad(lat))
    )
    area_1d = xr.DataArray(area_1d, coords={"lat": lat}, dims=("lat",))
    return area_1d * xr.ones_like(da.isel(time=0, drop=True))


def calc_global_annual_series(da):
    """Calculate area-integrated annual global wind work in TW."""
    area = compute_grid_area_2d(da)
    total_power_watts = (da * area).sum(dim=["lat", "lon"], skipna=True)
    annual_tw = (
        (total_power_watts / 1e12)
        .resample(time="1YE")
        .mean(dim="time", skipna=True)
        .load()
    )

    years = annual_tw["time"].dt.year.values.astype(int)
    values = annual_tw.values.astype(float)
    valid = np.isfinite(values)
    return years[valid], values[valid]


def calc_trend_stats(years, values):
    """Calculate the linear trend and the original 95% fitted-mean confidence band."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(years) & np.isfinite(values)
    x = years[valid]
    y = values[valid]

    n = len(x)
    if n < 3:
        raise ValueError("Not enough valid annual data points for trend calculation.")

    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    y_fit = slope * x + intercept

    t_score = stats.t.ppf(0.975, df=n - 2)
    mse = np.sum((y - y_fit) ** 2) / (n - 2)
    sx = np.sum((x - x.mean()) ** 2)
    ci = t_score * np.sqrt(mse * (1.0 / n + (x - x.mean()) ** 2 / sx))

    y_fit_all = slope * years + intercept
    ci_all = np.full_like(years, np.nan, dtype=float)
    ci_all[valid] = ci

    return {
        "slope_dec": float(slope * 10.0),
        "p_value": float(p_value),
        "y_fit": y_fit_all,
        "ci": ci_all,
        "mean": float(np.nanmean(y)),
        "start_pred": float(slope * YEAR_START + intercept),
        "end_pred": float(slope * YEAR_END + intercept),
    }


def format_p(p_value):
    """Format a P value for a figure legend."""
    if p_value < 0.001:
        return r"$P<0.001$"
    return rf"$P={p_value:.3f}$"


def get_robinson_y(latitudes):
    """Return Robinson-projection y coordinates for selected latitudes."""
    latitudes = np.asarray(latitudes, dtype=float)
    proj = ccrs.Robinson(central_longitude=180)
    x_coords = np.zeros_like(latitudes)
    pts = proj.transform_points(ccrs.PlateCarree(), x_coords, latitudes)
    return pts[:, 1]


def add_left_latitude_labels(ax, latitudes=(-60, -30, 0, 30, 60), fontsize=7.2):
    """Place latitude labels only along the left Robinson-map boundary."""
    latitudes = np.asarray(latitudes, dtype=float)
    seam_lon = np.zeros_like(latitudes)
    points = ax.projection.transform_points(ccrs.PlateCarree(), seam_lon, latitudes)

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


def cache_is_compatible(cache):
    """Check whether an existing plotting cache matches the current settings."""
    required_keys = {
        "cache_version",
        "year_start",
        "year_end",
        "diff_mode",
        "lat",
        "lon_cyclic",
        "mean_diff_cyclic_mw",
        "map_limit",
        "years",
        "series_corr",
        "series_slab",
        "corr_fit",
        "slab_fit",
        "corr_ci",
        "slab_ci",
        "corr_slope_dec",
        "slab_slope_dec",
        "corr_p",
        "slab_p",
    }

    if not required_keys.issubset(cache.files):
        return False

    cached_mode = str(cache["diff_mode"].item())
    return (
        int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and cached_mode == DIFF_MODE
    )


def compute_plot_data():
    """Compute all data needed to reproduce Supplementary Figure S1."""
    da_corr = open_power_input(CORR_FILE)
    da_slab = open_power_input(SLAB_FILE)
    da_corr, da_slab = xr.align(da_corr, da_slab, join="inner")

    if da_corr.sizes.get("time", 0) == 0:
        raise ValueError("Corrected and slab datasets have no overlapping time samples.")

    mean_corr = da_corr.mean(dim="time", skipna=True)
    mean_slab = da_slab.mean(dim="time", skipna=True)

    if DIFF_MODE == "corrected_minus_slab":
        mean_diff = mean_corr - mean_slab
    elif DIFF_MODE == "slab_minus_corrected":
        mean_diff = mean_slab - mean_corr
    else:
        raise ValueError(
            "DIFF_MODE must be 'corrected_minus_slab' or 'slab_minus_corrected'."
        )

    mean_diff_mw = mean_diff * 1000.0
    mean_diff_cyclic_mw, lon_cyclic = add_cyclic_point(
        mean_diff_mw.values,
        coord=mean_diff_mw["lon"].values,
    )

    valid_diff = np.abs(mean_diff_mw.values[np.isfinite(mean_diff_mw.values)])
    if valid_diff.size == 0:
        raise ValueError("The mean-difference map contains no finite values.")
    map_limit = float(np.nanpercentile(valid_diff, 99))
    if not np.isfinite(map_limit) or map_limit <= 0:
        map_limit = float(np.nanmax(valid_diff))
    if not np.isfinite(map_limit) or map_limit <= 0:
        map_limit = 1.0

    years_corr, series_corr = calc_global_annual_series(da_corr)
    years_slab, series_slab = calc_global_annual_series(da_slab)
    common_years = np.intersect1d(years_corr, years_slab)

    series_corr = series_corr[np.isin(years_corr, common_years)]
    series_slab = series_slab[np.isin(years_slab, common_years)]
    years = common_years.astype(int)

    stats_corr = calc_trend_stats(years, series_corr)
    stats_slab = calc_trend_stats(years, series_slab)

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "diff_mode": np.array(DIFF_MODE),
        "lat": mean_diff_mw["lat"].values.astype(float),
        "lon_cyclic": lon_cyclic.astype(float),
        "mean_diff_cyclic_mw": mean_diff_cyclic_mw.astype(float),
        "map_limit": np.array(map_limit, dtype=float),
        "years": years,
        "series_corr": series_corr.astype(float),
        "series_slab": series_slab.astype(float),
        "corr_fit": stats_corr["y_fit"].astype(float),
        "slab_fit": stats_slab["y_fit"].astype(float),
        "corr_ci": stats_corr["ci"].astype(float),
        "slab_ci": stats_slab["ci"].astype(float),
        "corr_slope_dec": np.array(stats_corr["slope_dec"], dtype=float),
        "slab_slope_dec": np.array(stats_slab["slope_dec"], dtype=float),
        "corr_p": np.array(stats_corr["p_value"], dtype=float),
        "slab_p": np.array(stats_slab["p_value"], dtype=float),
    }


def load_or_build_plot_data():
    """Load a compatible cache or rebuild it from the source datasets."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    missing = [path for path in (CORR_FILE, SLAB_FILE) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Supplementary Figure S1 cannot be generated because required source "
            "files are missing and no compatible plotting cache is available.\n\n"
            + "\n".join(str(path) for path in missing)
        )

    plot_data = compute_plot_data()
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Load or compute plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()

lat = data["lat"].astype(float)
lon_cyclic = data["lon_cyclic"].astype(float)
mean_diff_cyclic_mw = data["mean_diff_cyclic_mw"].astype(float)
map_limit = float(data["map_limit"])

years = data["years"].astype(int)
series_corr = data["series_corr"].astype(float)
series_slab = data["series_slab"].astype(float)
corr_fit = data["corr_fit"].astype(float)
slab_fit = data["slab_fit"].astype(float)
corr_ci = data["corr_ci"].astype(float)
slab_ci = data["slab_ci"].astype(float)
corr_slope_dec = float(data["corr_slope_dec"])
slab_slope_dec = float(data["slab_slope_dec"])
corr_p = float(data["corr_p"])
slab_p = float(data["slab_p"])


# =============================================================================
# 5. Supplementary Figure S1
# =============================================================================
# Width ≈ 183 mm, matching the main-figure format.
fig = plt.figure(figsize=(7.2, 8.0), facecolor="white")

pos_map = [0.08, 0.545, 0.84, 0.37]
pos_cbar = [0.29, 0.485, 0.42, 0.014]
pos_ts = [0.11, 0.095, 0.81, 0.305]

# Panel a: multidecadal mean wind-work difference.
ax_map = fig.add_axes(
    pos_map,
    projection=ccrs.Robinson(central_longitude=180),
)

map_image = ax_map.pcolormesh(
    lon_cyclic,
    lat,
    mean_diff_cyclic_mw,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-map_limit,
    vmax=map_limit,
    shading="auto",
    rasterized=True,
    zorder=1,
)

# Gridlines above the color field but below land/coastlines.
gridlines = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.7,
    color="#666666",
    alpha=0.45,
    linestyle="--",
    zorder=2,
)
gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
gridlines.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

ax_map.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=3)
ax_map.coastlines(linewidth=0.5, zorder=4)
ax_map.set_global()
ax_map.set_title("a", loc="left", fontweight="bold", pad=4)

# Separate invisible gridliner for bottom longitude labels only.
longitude_labels = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.0,
)
longitude_labels.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
longitude_labels.ylocator = mticker.FixedLocator([])
longitude_labels.top_labels = False
longitude_labels.bottom_labels = True
longitude_labels.left_labels = False
longitude_labels.right_labels = False
longitude_labels.xlabel_style = {"size": 7.2, "color": "#333333"}
if hasattr(longitude_labels, "geo_labels"):
    longitude_labels.geo_labels = False
try:
    longitude_labels.x_inline = False
except AttributeError:
    pass

add_left_latitude_labels(ax_map)

cbar_ax = fig.add_axes(pos_cbar)
colorbar = fig.colorbar(map_image, cax=cbar_ax, orientation="horizontal")
colorbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
colorbar.outline.set_linewidth(0.8)
fig.text(
    pos_cbar[0] + pos_cbar[2] + 0.012,
    pos_cbar[1] + 0.5 * pos_cbar[3],
    r"(mW m$^{-2}$)",
    ha="left",
    va="center",
    fontsize=8.0,
)

# Panel b: corrected and uncorrected global annual wind work.
ax_ts = fig.add_axes(pos_ts)

ax_ts.plot(
    years,
    series_corr,
    color="#D62728",
    linewidth=1.2,
    alpha=0.85,
    zorder=3,
    label="Corrected",
)
ax_ts.scatter(
    years,
    series_corr,
    color="#D62728",
    s=15,
    alpha=0.75,
    zorder=4,
)
ax_ts.plot(
    years,
    corr_fit,
    color="#D62728",
    linewidth=1.4,
    linestyle="--",
    zorder=5,
    label=(
        rf"Corrected trend: {corr_slope_dec * 1000:+.1f} GW decade$^{{-1}}$, "
        + format_p(corr_p)
    ),
)
ax_ts.fill_between(
    years,
    corr_fit - corr_ci,
    corr_fit + corr_ci,
    color="#D62728",
    alpha=0.13,
    linewidth=0,
    zorder=1,
)

ax_ts.plot(
    years,
    series_slab,
    color="#4D4D4D",
    linewidth=1.2,
    alpha=0.80,
    zorder=3,
    label="Uncorrected slab",
)
ax_ts.scatter(
    years,
    series_slab,
    color="#4D4D4D",
    s=15,
    alpha=0.70,
    zorder=4,
)
ax_ts.plot(
    years,
    slab_fit,
    color="#4D4D4D",
    linewidth=1.4,
    linestyle="--",
    zorder=5,
    label=(
        rf"Slab trend: {slab_slope_dec * 1000:+.1f} GW decade$^{{-1}}$, "
        + format_p(slab_p)
    ),
)
ax_ts.fill_between(
    years,
    slab_fit - slab_ci,
    slab_fit + slab_ci,
    color="#4D4D4D",
    alpha=0.11,
    linewidth=0,
    zorder=1,
)

ax_ts.set_title("b", loc="left", fontweight="bold", pad=4)
ax_ts.set_xlabel("Year")
ax_ts.set_ylabel("Global wind work (TW)")
ax_ts.set_xlim(YEAR_START - 1, YEAR_END + 1)
ax_ts.set_xticks(np.arange(1980, 2026, 5))

# Retain the original fixed y-axis range for direct comparability.
ax_ts.set_ylim(0.140, 0.240)
ax_ts.set_yticks(np.arange(0.14, 0.241, 0.02))
ax_ts.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    color="#B0B0B0",
    alpha=0.45,
)
ax_ts.legend(
    loc="lower right",
    frameon=False,
    handlelength=2.0,
    borderaxespad=0.3,
    labelspacing=0.35,
)
ax_ts.tick_params(width=0.8, length=3)


# =============================================================================
# 6. Save as a 400-dpi RGB PNG
# =============================================================================
fig.savefig(
    FIGURE_FILE,
    dpi=400,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)

# Matplotlib PNG output can contain an alpha channel; convert explicitly to RGB.
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
print(f"Plotting cache: {CACHE_FILE}")
