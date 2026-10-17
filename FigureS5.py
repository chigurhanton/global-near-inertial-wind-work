# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S7 from source data or a reusable cache.

Panels
------
a  Mixed-layer-depth trend.
b  Wind-work trend difference between interannually varying MLD and
   monthly climatological MLD experiments.
c  Global annual wind work for the two MLD experiments.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy import stats
from scipy.io import loadmat
from scipy.stats import linregress


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

VARYING_MLD_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
CLIMATOLOGICAL_MLD_FILE = DATA_DIR / "deeplearning_climmld_niw_power_input_1979_2025.nc"
MLD_FILE = DATA_DIR / "mld_all_1979to2025.mat"

CACHE_FILE = CACHE_DIR / "FigS5_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS5.png"

WINDWORK_VAR = "power_input"
MLD_VAR = "mld_all"
YEAR_START = 1979
YEAR_END = 2025
R_EARTH = 6_371_000.0

# Same spatial-analysis mask used in Figure 3.
MIN_ABS_LAT = 3.0
MAX_ABS_LAT = 60.0

# Cache version 2 records the use of a simple 12-month annual mean in panel c.
CACHE_VERSION = 2
RECOMPUTE_CACHE = False

# Spatial-map settings retained from the original analysis.
MAP_A_PERCENTILE = 98.0
MAP_B_PERCENTILE = 98.0
MLD_COLOR_LIMIT = None
WW_DIFF_COLOR_LIMIT = None
STIPPLE_STRIDE = 8
STIPPLE_SIZE = 0.65
STIPPLE_ALPHA = 0.55

COLOR_VARYING = "#D62728"
COLOR_CLIMATOLOGY = "#1F77B4"


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
# 3. Data utilities
# =============================================================================
def standardize_latlon_names(da):
    """Rename common latitude/longitude coordinate names to lat/lon."""
    rename = {}
    if "latitude" in da.coords or "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.coords or "longitude" in da.dims:
        rename["longitude"] = "lon"
    return da.rename(rename) if rename else da


def prepare_wind_work(ds, source_name):
    """Extract and standardize the requested wind-work field."""
    if WINDWORK_VAR not in ds.data_vars:
        raise KeyError(
            f"Variable '{WINDWORK_VAR}' was not found in {source_name}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = standardize_latlon_names(ds[WINDWORK_VAR])
    required = {"time", "lat", "lon"}
    available = set(da.coords) | set(da.dims)
    missing = required.difference(available)
    if missing:
        raise KeyError(f"Missing coordinates/dimensions in {source_name}: {sorted(missing)}")

    da = da.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
    da = da.sortby("lat").sortby("lon")
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"No wind-work data found during {YEAR_START}–{YEAR_END} in {source_name}.")
    return da


def read_mld_as_time_lat_lon(path):
    """Read mld_all using the same dimension-identification logic as the source code."""
    try:
        with h5py.File(path, "r") as handle:
            if MLD_VAR not in handle:
                raise KeyError(f"Variable '{MLD_VAR}' not found. Keys: {list(handle.keys())}")
            data = handle[MLD_VAR][()]
    except OSError:
        mat = loadmat(path)
        if MLD_VAR not in mat:
            available = [key for key in mat if not key.startswith("__")]
            raise KeyError(f"Variable '{MLD_VAR}' not found. Variables: {available}")
        data = mat[MLD_VAR]

    data = np.asarray(data, dtype=float).squeeze()
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D MLD array; obtained {data.shape}.")

    data[~np.isfinite(data)] = np.nan
    data[np.abs(data) > 1.0e4] = np.nan

    expected_months = (YEAR_END - YEAR_START + 1) * 12
    shape = data.shape

    time_candidates = [i for i, n in enumerate(shape) if n == expected_months]
    if not time_candidates:
        time_candidates = [i for i, n in enumerate(shape) if n >= expected_months]
    if not time_candidates:
        raise ValueError(
            f"Cannot identify the time axis in MLD shape {shape}; "
            f"expected approximately {expected_months} months."
        )
    axis_time = time_candidates[0]

    lon_candidates = [i for i, n in enumerate(shape) if n == 1440 and i != axis_time]
    if lon_candidates:
        axis_lon = lon_candidates[0]
    else:
        remaining = [i for i in range(3) if i != axis_time]
        axis_lon = max(remaining, key=lambda i: shape[i])

    axis_lat = [i for i in range(3) if i not in {axis_time, axis_lon}][0]
    data_tyl = np.transpose(data, (axis_time, axis_lat, axis_lon))
    ntime, nlat, nlon = data_tyl.shape

    time = pd.date_range(start=f"{YEAR_START}-01-01", periods=ntime, freq="MS")
    lat = np.linspace(-80.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)

    da = xr.DataArray(
        data_tyl,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="mld",
        attrs={"units": "m"},
    )
    return da.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")).sortby("lat").sortby("lon")


def harmonize_longitude_to_reference(da, reference):
    """Convert longitude convention to match a reference grid."""
    reference_uses_negative = float(reference["lon"].min()) < 0.0
    data_uses_negative = float(da["lon"].min()) < 0.0

    if reference_uses_negative and not data_uses_negative:
        da = da.assign_coords(lon=((da["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")
    elif not reference_uses_negative and data_uses_negative:
        da = da.assign_coords(lon=da["lon"] % 360.0).sortby("lon")
    return da


def apply_analysis_mask(da):
    """Apply the 3°–60° latitude mask used for the spatial diagnostics."""
    mask = (np.abs(da["lat"]) >= MIN_ABS_LAT) & (np.abs(da["lat"]) <= MAX_ABS_LAT)
    return da.where(mask)


def annual_mean_field(da):
    """Simple arithmetic mean of monthly fields within each calendar year."""
    return da.resample(time="1YE").mean(dim="time", skipna=True)


def calc_spatial_trend_and_pvalue(annual_da):
    """Pixel-wise linear trend and P value; slope is returned per year."""
    def _linear_trend(y):
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 5:
            return np.nan, np.nan
        x = np.arange(y.size, dtype=float)
        slope, _, _, p_value, _ = linregress(x[valid], y[valid])
        return slope, p_value

    slope, p_value = xr.apply_ufunc(
        _linear_trend,
        annual_da,
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    return slope.load(), p_value.load()


def compute_grid_area_2d(da):
    """Compute regular lat-lon grid-cell area in m²."""
    lat = da["lat"]
    lon = da["lon"]
    dlat = float(np.abs(lat[1] - lat[0]))
    dlon = float(np.abs(lon[1] - lon[0]))

    area_1d = (
        R_EARTH**2
        * np.deg2rad(dlat)
        * np.deg2rad(dlon)
        * np.cos(np.deg2rad(lat))
    )
    area_1d = xr.DataArray(area_1d, coords={"lat": lat}, dims=("lat",))
    return area_1d * xr.ones_like(da.isel(time=0, drop=True))


def calc_global_annual_series(da):
    """Area-integrate monthly wind work, then take a simple 12-month annual mean."""
    area = compute_grid_area_2d(da)
    monthly_global_tw = (da * area).sum(dim=["lat", "lon"], skipna=True) / 1.0e12
    annual_tw = monthly_global_tw.groupby("time.year").mean(dim="time", skipna=True).load()

    years = annual_tw["year"].values.astype(int)
    values = annual_tw.values.astype(float)
    valid = np.isfinite(values)
    return years[valid], values[valid]


def calc_time_series_trend(years, values):
    """Linear trend and 95% fitted-line confidence interval."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(years) & np.isfinite(values)
    x = years[valid]
    y = values[valid]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    fitted = slope * x + intercept

    n = x.size
    t_score = stats.t.ppf(0.975, df=n - 2)
    mse = np.sum((y - fitted) ** 2) / (n - 2)
    sx = np.sum((x - x.mean()) ** 2)
    fit_ci_valid = t_score * np.sqrt(mse * (1.0 / n + (x - x.mean()) ** 2 / sx))

    fit_all = slope * years + intercept
    fit_ci_all = np.full(years.shape, np.nan, dtype=float)
    fit_ci_all[valid] = fit_ci_valid

    return {
        "slope_decade": slope * 10.0,
        "p_value": p_value,
        "mean": np.nanmean(y),
        "y_fit": fit_all,
        "fit_ci": fit_ci_all,
        "r_squared": r_value**2,
        "slope_ci_decade": t_score * std_err * 10.0,
    }


# =============================================================================
# 4. Cache generation
# =============================================================================
def cache_is_compatible(cache):
    """Check whether an existing S7 plotting cache matches current settings."""
    required = {
        "cache_version",
        "year_start",
        "year_end",
        "min_abs_lat",
        "max_abs_lat",
        "lat",
        "lon",
        "mld_trend_decade",
        "pvalue_mld",
        "windwork_trend_difference_decade",
        "pvalue_difference",
        "years",
        "series_varying",
        "series_climatology",
        "fit_varying",
        "fit_ci_varying",
        "fit_climatology",
        "fit_ci_climatology",
        "slope_varying_gw_dec",
        "slope_climatology_gw_dec",
        "p_varying",
        "p_climatology",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and np.isclose(float(cache["min_abs_lat"]), MIN_ABS_LAT)
        and np.isclose(float(cache["max_abs_lat"]), MAX_ABS_LAT)
    )


def compute_plot_data():
    """Compute all fields and annual series needed to draw Figure S7."""
    missing = [path for path in [VARYING_MLD_FILE, CLIMATOLOGICAL_MLD_FILE, MLD_FILE] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure S7 source data are missing:\n\n" + "\n".join(f"  {path}" for path in missing)
        )

    with xr.open_dataset(VARYING_MLD_FILE) as ds_varying, xr.open_dataset(CLIMATOLOGICAL_MLD_FILE) as ds_clim:
        varying = prepare_wind_work(ds_varying, VARYING_MLD_FILE.name)
        climatology = prepare_wind_work(ds_clim, CLIMATOLOGICAL_MLD_FILE.name)
        varying, climatology = xr.align(varying, climatology, join="inner")

        if varying.size == 0 or climatology.size == 0:
            raise ValueError("The two wind-work experiments have no overlapping data grid.")

        # Spatial diagnostics use annual arithmetic-mean fields.
        varying_annual = annual_mean_field(varying)
        climatology_annual = annual_mean_field(climatology)
        varying_annual, climatology_annual = xr.align(varying_annual, climatology_annual, join="inner")
        annual_difference = varying_annual - climatology_annual

        # ORAS5 MLD is prepared on the wind-work grid using the original nearest-neighbour method.
        mld = read_mld_as_time_lat_lon(MLD_FILE)
        mld_annual = annual_mean_field(mld)
        reference_grid = varying_annual.isel(time=0, drop=True)
        mld_annual = harmonize_longitude_to_reference(mld_annual, reference_grid)
        mld_annual = mld_annual.interp_like(reference_grid, method="nearest")

        slope_mld_year, pvalue_mld = calc_spatial_trend_and_pvalue(mld_annual)
        slope_diff_year, pvalue_difference = calc_spatial_trend_and_pvalue(annual_difference)

        mld_trend_decade = apply_analysis_mask(slope_mld_year * 10.0)
        pvalue_mld = apply_analysis_mask(pvalue_mld)
        ww_trend_difference_decade = apply_analysis_mask(slope_diff_year * 10.0 * 1000.0)
        pvalue_difference = apply_analysis_mask(pvalue_difference)

        # Panel c uses a simple average of the 12 monthly global values in each year.
        years_varying, series_varying = calc_global_annual_series(varying)
        years_clim, series_climatology = calc_global_annual_series(climatology)

    common_years = np.intersect1d(years_varying, years_clim)
    varying_lookup = dict(zip(years_varying, series_varying))
    clim_lookup = dict(zip(years_clim, series_climatology))

    years = common_years.astype(int)
    series_varying = np.array([varying_lookup[year] for year in years], dtype=float)
    series_climatology = np.array([clim_lookup[year] for year in years], dtype=float)

    stats_varying = calc_time_series_trend(years, series_varying)
    stats_climatology = calc_time_series_trend(years, series_climatology)

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "min_abs_lat": np.array(MIN_ABS_LAT, dtype=float),
        "max_abs_lat": np.array(MAX_ABS_LAT, dtype=float),
        "lat": mld_trend_decade["lat"].values,
        "lon": mld_trend_decade["lon"].values,
        "mld_trend_decade": mld_trend_decade.values,
        "pvalue_mld": pvalue_mld.values,
        "windwork_trend_difference_decade": ww_trend_difference_decade.values,
        "pvalue_difference": pvalue_difference.values,
        "years": years,
        "series_varying": series_varying,
        "series_climatology": series_climatology,
        "fit_varying": stats_varying["y_fit"],
        "fit_ci_varying": stats_varying["fit_ci"],
        "fit_climatology": stats_climatology["y_fit"],
        "fit_ci_climatology": stats_climatology["fit_ci"],
        "slope_varying_gw_dec": np.array(stats_varying["slope_decade"] * 1000.0, dtype=float),
        "slope_climatology_gw_dec": np.array(stats_climatology["slope_decade"] * 1000.0, dtype=float),
        "p_varying": np.array(stats_varying["p_value"], dtype=float),
        "p_climatology": np.array(stats_climatology["p_value"], dtype=float),
    }


def load_or_build_plot_data():
    """Load a compatible cache or calculate it from source data."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    plot_data = compute_plot_data()
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 5. Load plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()
lat = data["lat"]
lon = data["lon"]
coords = {"lat": lat, "lon": lon}

mld_trend = xr.DataArray(data["mld_trend_decade"], coords=coords, dims=("lat", "lon"))
pvalue_mld = xr.DataArray(data["pvalue_mld"], coords=coords, dims=("lat", "lon"))
ww_trend_difference = xr.DataArray(
    data["windwork_trend_difference_decade"], coords=coords, dims=("lat", "lon")
)
pvalue_difference = xr.DataArray(data["pvalue_difference"], coords=coords, dims=("lat", "lon"))

years = data["years"].astype(int)
series_varying = data["series_varying"].astype(float)
series_climatology = data["series_climatology"].astype(float)
fit_varying = data["fit_varying"].astype(float)
fit_ci_varying = data["fit_ci_varying"].astype(float)
fit_climatology = data["fit_climatology"].astype(float)
fit_ci_climatology = data["fit_ci_climatology"].astype(float)
slope_varying = float(data["slope_varying_gw_dec"])
slope_climatology = float(data["slope_climatology_gw_dec"])
p_varying = float(data["p_varying"])
p_climatology = float(data["p_climatology"])


# =============================================================================
# 6. Plot helpers
# =============================================================================
def format_p(p_value):
    return r"$P<0.001$" if p_value < 0.001 else rf"$P={p_value:.3f}$"


def robust_symmetric_limit(da, percentile):
    values = np.asarray(da.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


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


def add_map_gridlines(ax, show_bottom_labels):
    """Draw gridlines without automatic latitude labels."""
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
        longitude_labels = ax.gridlines(
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

    add_left_latitude_labels(ax)


def add_stippling(ax, pvalue_da):
    """Draw spatially thinned P<0.05 stippling."""
    pvalues = np.asarray(pvalue_da.values, dtype=float)
    significant = np.isfinite(pvalues) & (pvalues < 0.05)
    significant_sub = significant[::STIPPLE_STRIDE, ::STIPPLE_STRIDE]
    if not np.any(significant_sub):
        return

    lon_sub = pvalue_da["lon"].values[::STIPPLE_STRIDE]
    lat_sub = pvalue_da["lat"].values[::STIPPLE_STRIDE]
    lon_2d, lat_2d = np.meshgrid(lon_sub, lat_sub)

    ax.scatter(
        lon_2d[significant_sub],
        lat_2d[significant_sub],
        s=STIPPLE_SIZE,
        color="black",
        marker=".",
        linewidths=0,
        alpha=STIPPLE_ALPHA,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=3,
    )


def plot_map(ax, da, pvalue_da, panel_label, cmap, percentile, manual_limit, show_bottom):
    """Plot one Robinson trend map with significance stippling."""
    cyclic_data, cyclic_lon = add_cyclic_point(da.values, coord=da["lon"].values)
    vmax = float(manual_limit) if manual_limit is not None else robust_symmetric_limit(da, percentile)

    image = ax.pcolormesh(
        cyclic_lon,
        da["lat"].values,
        cyclic_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
        rasterized=True,
        zorder=1,
    )

    add_stippling(ax, pvalue_da)
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=5)
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=8)
    ax.set_global()
    add_map_gridlines(ax, show_bottom)
    ax.set_title(panel_label, loc="left", fontweight="bold", pad=4)
    return image


def automatic_y_axis(*arrays):
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    data_min = float(values.min())
    data_max = float(values.max())
    data_range = data_max - data_min
    padding = max(0.003, 0.07 * data_range)

    if data_range <= 0.04:
        step = 0.01
    elif data_range <= 0.08:
        step = 0.02
    else:
        step = 0.025

    lower = np.floor((data_min - padding) / step) * step
    upper = np.ceil((data_max + padding) / step) * step
    ticks = np.arange(lower, upper + 0.5 * step, step)
    return lower, upper, ticks


# =============================================================================
# 7. Figure S7
# =============================================================================
fig = plt.figure(figsize=(6.4, 7.55), facecolor="white")

# Manual axes positions keep the two map/colorbar groups compact. Panel c is
# matched to the actual Robinson map width after Cartopy resolves projection geometry.
ax_a = fig.add_axes([0.075, 0.710, 0.85, 0.225], projection=ccrs.Robinson(central_longitude=180))
ax_b = fig.add_axes([0.075, 0.405, 0.85, 0.225], projection=ccrs.Robinson(central_longitude=180))
ax_c = fig.add_axes([0.125, 0.080, 0.76, 0.235])

# Map colorbars are deliberately shorter than the maps.
cax_a = fig.add_axes([0.315, 0.675, 0.37, 0.015])
cax_b = fig.add_axes([0.315, 0.355, 0.37, 0.015])

im_a = plot_map(
    ax_a,
    mld_trend,
    pvalue_mld,
    "a",
    "PiYG_r",
    MAP_A_PERCENTILE,
    MLD_COLOR_LIMIT,
    show_bottom=False,
)
cb_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal")
cb_a.ax.tick_params(labelsize=7.2, width=0.8, length=3)
cb_a.outline.set_linewidth(0.8)
cb_a.ax.text(
    1.03,
    0.5,
    r"(m decade$^{-1}$)",
    transform=cb_a.ax.transAxes,
    ha="left",
    va="center",
    fontsize=8.0,
)

im_b = plot_map(
    ax_b,
    ww_trend_difference,
    pvalue_difference,
    "b",
    "RdBu_r",
    MAP_B_PERCENTILE,
    WW_DIFF_COLOR_LIMIT,
    show_bottom=True,
)
cb_b = fig.colorbar(im_b, cax=cax_b, orientation="horizontal")
cb_b.ax.tick_params(labelsize=7.2, width=0.8, length=3)
cb_b.outline.set_linewidth(0.8)
cb_b.ax.text(
    1.03,
    0.5,
    r"(mW m$^{-2}$ decade$^{-1}$)",
    transform=cb_b.ax.transAxes,
    ha="left",
    va="center",
    fontsize=8.0,
)

# Panel c: global annual wind work.
ax_c.plot(
    years,
    series_varying,
    color=COLOR_VARYING,
    linewidth=1.2,
    marker="o",
    markersize=3.0,
    alpha=0.84,
    label="Interannually varying MLD",
)
ax_c.plot(
    years,
    fit_varying,
    color=COLOR_VARYING,
    linewidth=1.4,
    linestyle="--",
    label=rf"Varying-MLD trend: {slope_varying:+.1f} GW decade$^{{-1}}$, {format_p(p_varying)}",
)
ax_c.plot(
    years,
    series_climatology,
    color=COLOR_CLIMATOLOGY,
    linewidth=1.2,
    marker="o",
    markersize=3.0,
    alpha=0.82,
    label="Monthly climatological MLD",
)
ax_c.plot(
    years,
    fit_climatology,
    color=COLOR_CLIMATOLOGY,
    linewidth=1.4,
    linestyle="--",
    label=rf"Climatological-MLD trend: {slope_climatology:+.1f} GW decade$^{{-1}}$, {format_p(p_climatology)}",
)

ax_c.set_title("c", loc="left", fontweight="bold", pad=4)
ax_c.set_xlabel("Year")
ax_c.set_ylabel("Global wind work (TW)")
ax_c.set_xlim(years.min() - 1, years.max() + 1)
ax_c.set_xticks(np.arange(1980, 2026, 5))

ymin, ymax, yticks = automatic_y_axis(
    series_varying,
    series_climatology,
    fit_varying - fit_ci_varying,
    fit_varying + fit_ci_varying,
    fit_climatology - fit_ci_climatology,
    fit_climatology + fit_ci_climatology,
)
ax_c.set_ylim(ymin, ymax)
ax_c.set_yticks(yticks)
ax_c.grid(True, linestyle="--", linewidth=0.7, color="#B0B0B0", alpha=0.35)
ax_c.tick_params(width=0.8, length=3)
ax_c.legend(
    frameon=False,
    loc="lower right",
    fontsize=6.0,
    handlelength=1.8,
    borderaxespad=0.35,
    labelspacing=0.28,
)

# Cartopy determines the final Robinson map width only after a render pass.
# Match panel c exactly to the common visible width of panels a and b.
fig.canvas.draw()
map_a_pos = ax_a.get_position(original=False)
map_b_pos = ax_b.get_position(original=False)
panel_c_pos = ax_c.get_position(original=False)

map_left = max(map_a_pos.x0, map_b_pos.x0)
map_right = min(map_a_pos.x1, map_b_pos.x1)
map_width = map_right - map_left
if map_width <= 0:
    raise RuntimeError("Failed to determine a valid common Robinson map width.")

ax_c.set_position([
    map_left,
    panel_c_pos.y0,
    map_width,
    panel_c_pos.height,
])

# Save as a 400-dpi RGB PNG.
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
print(f"Plotting cache: {CACHE_FILE}")
