# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S2 from source data or a reusable cache.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt


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

WIND_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
CACHE_FILE = CACHE_DIR / "FigS2_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS2.png"

POWER_VAR = "power_input"
YEAR_START = 1979
YEAR_END = 2025
HAC_LAG = 3
R_EARTH = 6_371_000.0
CACHE_VERSION = 1
RECOMPUTE_CACHE = False

# Starting years retained from the original regional-trend analysis.
START_YEARS = np.array([1979, 1990, 1992, 1994, 1996, 1998, 2000], dtype=int)

# Regional definitions. Longitudes are represented on a 0–360° grid.
# North Pacific: 0–60°N, 100°E–100°W (260°E).
NP_LAT_MIN = 0.0
NP_LAT_MAX = 60.0
NP_LON_MIN = 100.0
NP_LON_MAX = 260.0

# North Atlantic: 0–60°N, 100°W–20°E.
NA_LAT_MIN = 0.0
NA_LAT_MAX = 60.0

# Southern Ocean: 40–60°S, all longitudes.
SO_LAT_MIN = -60.0
SO_LAT_MAX = -40.0

# Original regional colors.
COLOR_NP = "#1F77B4"
COLOR_NP_RECENT = "#08306B"
COLOR_NA = "#E67E22"
COLOR_SO = "#2CA02C"


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
# 3. Helper functions
# =============================================================================
def find_coord(ds, candidates):
    """Return the first matching coordinate or dimension name."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Cannot identify coordinate from {candidates}")


def calc_trend_hac(years, values, hac_lag=HAC_LAG):
    """Fit an OLS trend and return HAC-robust significance statistics."""
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(years) & np.isfinite(values)
    x = years[valid]
    y = values[valid]

    if x.size < 4:
        raise ValueError("At least four annual values are required for trend estimation.")

    x_mean = float(np.mean(x))
    xc = x - x_mean

    design = sm.add_constant(xc)
    ols = sm.OLS(y, design).fit()
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=min(hac_lag, len(y) - 1),
    )

    # TW year^-1 -> GW decade^-1.
    trend_gw_dec = float(ols.params[1] * 10_000.0)
    p_hac = float(hac.pvalues[1])

    xfit = np.linspace(np.min(x), np.max(x), 300)
    yfit = ols.params[0] + ols.params[1] * (xfit - x_mean)

    return {
        "trend": trend_gw_dec,
        "p": p_hac,
        "xfit": xfit,
        "yfit": yfit,
    }


def format_p(p_value):
    """Format P values for figure legends."""
    if p_value < 0.001:
        return r"$P<0.001$"
    return rf"$P={p_value:.3f}$"


def trend_linestyle(p_value):
    """Use a solid line for significant trends and dashed otherwise."""
    return "-" if p_value < 0.05 else "--"


def integrate_region(da, area, lat_name, lon_name, mask=None):
    """Area-integrate wind work over a regional mask and return TW."""
    field = da if mask is None else da.where(mask)
    return (
        (field * area)
        .sum(dim=[lat_name, lon_name], skipna=True)
        / 1e12
    )


def cache_is_compatible(cache):
    """Check whether an existing plotting cache matches current settings."""
    required = {
        "cache_version",
        "year_start",
        "year_end",
        "hac_lag",
        "years",
        "global_tw",
        "north_pacific_tw",
        "north_atlantic_tw",
        "southern_ocean_tw",
        "start_years",
        "trend_table",
        "p_table",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and int(cache["hac_lag"]) == HAC_LAG
        and np.array_equal(cache["start_years"], START_YEARS)
    )


def compute_plot_data(source_file):
    """Compute annual regional series and trend statistics from the source data."""
    with xr.open_dataset(source_file, decode_times=True) as ds:
        time_name = find_coord(ds, ["time", "Time", "TIME"])
        lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT"])
        lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON"])

        if POWER_VAR not in ds:
            raise KeyError(f"Variable '{POWER_VAR}' was not found in {source_file.name}.")

        ds = ds.sel({
            time_name: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")
        })

        lon360 = xr.where(ds[lon_name] < 0, ds[lon_name] + 360.0, ds[lon_name])
        ds = (
            ds.assign_coords({lon_name: lon360})
            .sortby(lon_name)
            .sortby(lat_name)
        )

        wind_work = (
            ds[POWER_VAR]
            .squeeze(drop=True)
            .transpose(time_name, lat_name, lon_name)
        )

        lat = ds[lat_name]
        lon = ds[lon_name]

        dlat = float(np.nanmedian(np.abs(np.diff(lat.values.astype(float)))))
        dlon = float(np.nanmedian(np.abs(np.diff(lon.values.astype(float)))))
        area = (
            R_EARTH**2
            * np.deg2rad(dlat)
            * np.deg2rad(dlon)
            * np.cos(np.deg2rad(lat))
        )

        mask_np = (
            (lat >= NP_LAT_MIN)
            & (lat <= NP_LAT_MAX)
            & (lon >= NP_LON_MIN)
            & (lon <= NP_LON_MAX)
        )
        mask_na = (
            (lat >= NA_LAT_MIN)
            & (lat <= NA_LAT_MAX)
            & ((lon >= 260.0) | (lon <= 20.0))
        )
        mask_so = (lat >= SO_LAT_MIN) & (lat <= SO_LAT_MAX)

        global_monthly = integrate_region(wind_work, area, lat_name, lon_name).load()
        np_monthly = integrate_region(wind_work, area, lat_name, lon_name, mask_np).load()
        na_monthly = integrate_region(wind_work, area, lat_name, lon_name, mask_na).load()
        so_monthly = integrate_region(wind_work, area, lat_name, lon_name, mask_so).load()

        monthly = pd.DataFrame({
            "time": pd.to_datetime(wind_work[time_name].values),
            "Global": np.asarray(global_monthly.values).squeeze(),
            "North_Pacific": np.asarray(np_monthly.values).squeeze(),
            "North_Atlantic": np.asarray(na_monthly.values).squeeze(),
            "Southern_Ocean": np.asarray(so_monthly.values).squeeze(),
        })

    monthly["year"] = monthly["time"].dt.year
    annual = (
        monthly.groupby("year")[[
            "Global",
            "North_Pacific",
            "North_Atlantic",
            "Southern_Ocean",
        ]]
        .mean()
        .loc[YEAR_START:YEAR_END]
    )

    region_columns = ["Global", "North_Pacific", "North_Atlantic", "Southern_Ocean"]
    trend_table = np.full((len(START_YEARS), len(region_columns)), np.nan, dtype=float)
    p_table = np.full_like(trend_table, np.nan)

    for i, start_year in enumerate(START_YEARS):
        subset = annual.loc[int(start_year):YEAR_END]
        for j, column in enumerate(region_columns):
            result = calc_trend_hac(subset.index.values, subset[column].values)
            trend_table[i, j] = result["trend"]
            p_table[i, j] = result["p"]

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "hac_lag": np.array(HAC_LAG, dtype=np.int16),
        "years": annual.index.to_numpy(dtype=int),
        "global_tw": annual["Global"].to_numpy(dtype=float),
        "north_pacific_tw": annual["North_Pacific"].to_numpy(dtype=float),
        "north_atlantic_tw": annual["North_Atlantic"].to_numpy(dtype=float),
        "southern_ocean_tw": annual["Southern_Ocean"].to_numpy(dtype=float),
        "start_years": START_YEARS,
        "trend_table": trend_table,
        "p_table": p_table,
    }


def load_or_build_plot_data():
    """Load a compatible cache or rebuild it from the source NetCDF file."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    if not WIND_FILE.exists():
        raise FileNotFoundError(
            "Supplementary Figure S2 cannot be generated because neither a compatible "
            "plotting cache nor the source NetCDF file is available.\n\n"
            f"Expected plotting cache:\n  {CACHE_FILE}\n\n"
            f"Expected source data for the first run:\n  {WIND_FILE}\n\n"
            "Place the source NetCDF file in the project 'data' folder for the first "
            "run, or place a compatible FigS2_plot_data.npz file in 'cache'."
        )

    plot_data = compute_plot_data(WIND_FILE)
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Load plotting data and calculate fitted trends
# =============================================================================
data, data_source = load_or_build_plot_data()

years = data["years"].astype(int)
series_np = data["north_pacific_tw"].astype(float)
series_na = data["north_atlantic_tw"].astype(float)
series_so = data["southern_ocean_tw"].astype(float)

np_full = calc_trend_hac(years, series_np)
recent_mask = years >= 1995
np_recent = calc_trend_hac(years[recent_mask], series_np[recent_mask])
na_full = calc_trend_hac(years, series_na)
so_full = calc_trend_hac(years, series_so)


# =============================================================================
# 5. Supplementary Figure S2
# =============================================================================
# Compact 1 x 3 layout at the same 18.3-cm journal width used for the other
# finalized figures. Each panel keeps its own y-axis ticks, while the y-axis
# title is shown only once in panel a.
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 3.2),
    sharex=True,
    facecolor="white",
)
plt.subplots_adjust(
    left=0.085,
    right=0.992,
    top=0.905,
    bottom=0.180,
    wspace=0.22,
)


def plot_series(
    ax,
    values,
    color,
    panel_label,
    trend_results,
    second_trend=None,
    show_ylabel=False,
):
    """Plot one regional annual series with one or two fitted trends."""
    ax.plot(
        years,
        values,
        color=color,
        linewidth=1.2,
        marker="o",
        markersize=2.8,
        alpha=0.88,
        zorder=3,
    )

    label_main = (
        rf"1979–2025: {trend_results['trend']:+.2f} GW decade$^{{-1}}$"
        + "\n"
        + format_p(trend_results["p"])
    )
    ax.plot(
        trend_results["xfit"],
        trend_results["yfit"],
        color=color,
        linestyle=trend_linestyle(trend_results["p"]),
        linewidth=1.4,
        label=label_main,
        zorder=5,
    )

    if second_trend is not None:
        label_recent = (
            rf"1995–2025: {second_trend['trend']:+.2f} GW decade$^{{-1}}$"
            + "\n"
            + format_p(second_trend["p"])
        )
        ax.plot(
            second_trend["xfit"],
            second_trend["yfit"],
            color=COLOR_NP_RECENT,
            linestyle=trend_linestyle(second_trend["p"]),
            linewidth=1.4,
            label=label_recent,
            zorder=6,
        )

    ax.set_title(panel_label, loc="left", fontweight="bold", pad=3)
    ax.set_ylabel("Wind work (TW)" if show_ylabel else "")
    ax.set_xlabel("Year")
    ax.set_xlim(YEAR_START - 1, YEAR_END + 1)

    # A 10-year label interval is used in the compact 1 x 3 layout so that
    # all three panels retain readable year labels at final print size.
    ax.set_xticks(np.arange(1980, 2021, 10))

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        color="#B0B0B0",
        alpha=0.35,
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1, 0.02),
        frameon=False,
        handlelength=1.5,
        borderaxespad=0.15,
        labelspacing=0.14,
        handletextpad=0.30,
        fontsize=5.8,
    )
    ax.tick_params(width=0.8, length=3)


plot_series(
    axes[0],
    series_np,
    COLOR_NP,
    "a",
    np_full,
    second_trend=np_recent,
    show_ylabel=True,
)
plot_series(axes[1], series_na, COLOR_NA, "b", na_full)
plot_series(axes[2], series_so, COLOR_SO, "c", so_full)


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
