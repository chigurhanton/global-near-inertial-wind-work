# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S3 from source data or a reusable cache.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
NINO_FILE = DATA_DIR / "nino34.long.anom.nc"
CACHE_FILE = CACHE_DIR / "FigS3_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS3.png"

POWER_VAR = "power_input"
NINO_VAR_CANDIDATES = ["value", "nino34", "NINO34", "anom", "sst"]

YEAR_START = 1979
YEAR_END = 2025
R_EARTH = 6_371_000.0
TW_TO_GW = 1000.0
CACHE_VERSION = 3
RECOMPUTE_CACHE = False

# North Pacific region used in the original calculation:
# 0–60°N, 100°E–100°W (260°E on a 0–360° longitude grid).
LAT_MIN = 0.0
LAT_MAX = 60.0
LON_MIN = 100.0
LON_MAX = 260.0


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
def find_coord_name(ds, candidates):
    """Return the first matching coordinate, dimension, or 1-D variable name."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
        if name in ds.variables and ds[name].ndim == 1:
            return name
    raise KeyError(f"Cannot find coordinate among: {candidates}")


def find_data_var(ds, preferred_names=None):
    """Return a preferred data variable, or the first available data variable."""
    preferred_names = preferred_names or []
    for name in preferred_names:
        if name in ds.data_vars:
            return name
    for name in ds.data_vars:
        return name
    raise KeyError("No suitable data variable found.")


def standardize_longitude(da, lon_name):
    """Convert longitude to 0–360° and sort it in ascending order."""
    lon360 = xr.where(da[lon_name] < 0, da[lon_name] + 360.0, da[lon_name])
    return da.assign_coords({lon_name: lon360}).sortby(lon_name)


def infer_lat_lon_time_names(ds):
    """Identify latitude, longitude, and time coordinate names."""
    lat_name = find_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = find_coord_name(ds, ["lon", "longitude", "x"])
    time_name = find_coord_name(ds, ["time", "date"])
    return lat_name, lon_name, time_name


def compute_lon_edges(lon_1d):
    """Return cell-edge longitudes for a regular 1-D longitude coordinate."""
    lon = np.asarray(lon_1d, dtype=float)
    if lon.size < 2:
        raise ValueError("Longitude coordinate must contain at least two points.")

    dlon = np.diff(lon)
    dlon_mean = float(np.median(dlon))

    edges = np.empty(lon.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lon[:-1] + lon[1:])
    edges[0] = lon[0] - dlon_mean / 2.0
    edges[-1] = lon[-1] + dlon_mean / 2.0
    return edges


def compute_lat_edges(lat_1d):
    """Return cell-edge latitudes for a regular 1-D latitude coordinate."""
    lat = np.asarray(lat_1d, dtype=float)
    if lat.size < 2:
        raise ValueError("Latitude coordinate must contain at least two points.")

    dlat = np.diff(lat)
    dlat_mean = float(np.median(np.abs(dlat)))

    edges = np.empty(lat.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    edges[0] = lat[0] - dlat_mean / 2.0
    edges[-1] = lat[-1] + dlat_mean / 2.0
    return np.clip(edges, -90.0, 90.0)


def compute_grid_area(lat_1d, lon_1d):
    """Compute spherical grid-cell area (m²) for a regular latitude-longitude grid."""
    lat_edges = np.deg2rad(compute_lat_edges(lat_1d))
    lon_edges = np.deg2rad(compute_lon_edges(lon_1d))

    dlon = np.diff(lon_edges)
    dphi = np.diff(np.sin(lat_edges))
    area = (R_EARTH**2) * dphi[:, None] * dlon[None, :]

    return xr.DataArray(
        area,
        coords={"lat_tmp": lat_1d, "lon_tmp": lon_1d},
        dims=("lat_tmp", "lon_tmp"),
    )


def simple_annual_mean_from_monthly(series, time_name="time"):
    """Convert monthly values to annual values using a simple 12-month mean."""
    return series.groupby(f"{time_name}.year").mean(dim=time_name, skipna=True)


def format_p(p_value):
    """Format a P value for the correlation annotation."""
    if p_value < 0.001:
        return r"$P<0.001$"
    return rf"$P={p_value:.3f}$"


def symmetric_ylim(data, pad=0.10):
    """Return symmetric limits around zero so twin-axis zeros align exactly."""
    data = np.asarray(data, dtype=float)
    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmax *= 1.0 + pad
    return -vmax, vmax


def cache_is_compatible(cache):
    """Check whether an existing plotting cache matches the current settings."""
    required = {
        "cache_version",
        "year_start",
        "year_end",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "years",
        "np_wind_work_detrended_anomaly_gw",
        "nino34_annual",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and np.isclose(float(cache["lat_min"]), LAT_MIN)
        and np.isclose(float(cache["lat_max"]), LAT_MAX)
        and np.isclose(float(cache["lon_min"]), LON_MIN)
        and np.isclose(float(cache["lon_max"]), LON_MAX)
    )


def compute_plot_data(wind_file, nino_file):
    """Compute detrended annual North Pacific wind-work and Niño 3.4 anomalies."""
    if not wind_file.exists():
        raise FileNotFoundError(f"Wind-work source file not found: {wind_file}")
    if not nino_file.exists():
        raise FileNotFoundError(f"Niño 3.4 source file not found: {nino_file}")

    with xr.open_dataset(wind_file, decode_times=True) as ds_wind:
        lat_name, lon_name, time_name = infer_lat_lon_time_names(ds_wind)

        wind_var = POWER_VAR if POWER_VAR in ds_wind.data_vars else find_data_var(
            ds_wind, preferred_names=[POWER_VAR]
        )
        wind_da = ds_wind[wind_var].sel(
            {time_name: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")}
        )
        wind_da = standardize_longitude(wind_da, lon_name)
        if wind_da[lat_name].values[0] > wind_da[lat_name].values[-1]:
            wind_da = wind_da.sortby(lat_name)

        wind_np = wind_da.sel({
            lat_name: slice(LAT_MIN, LAT_MAX),
            lon_name: slice(LON_MIN, LON_MAX),
        })

        lat_vals = wind_np[lat_name].values
        lon_vals = wind_np[lon_name].values
        area_2d = compute_grid_area(lat_vals, lon_vals).rename(
            {"lat_tmp": lat_name, "lon_tmp": lon_name}
        )

        monthly_np_tw = (
            (wind_np * area_2d).sum(dim=(lat_name, lon_name), skipna=True) / 1e12
        )
        annual_np_tw = simple_annual_mean_from_monthly(
            monthly_np_tw,
            time_name=time_name,
        ).sel(year=slice(YEAR_START, YEAR_END))
        annual_np_tw = annual_np_tw.load()

    # Detrend the 1979–2025 annual North Pacific wind-work series.
    # The anomaly is defined as the residual about the fitted linear background.
    np_years = annual_np_tw["year"].values.astype(float)
    np_values_tw = annual_np_tw.values.astype(float)

    valid_np = np.isfinite(np_years) & np.isfinite(np_values_tw)
    if np.count_nonzero(valid_np) < 3:
        raise ValueError(
            "Not enough valid annual North Pacific wind-work values for detrending."
        )

    np_slope_tw_per_year, np_intercept_tw, _, _, _ = stats.linregress(
        np_years[valid_np],
        np_values_tw[valid_np],
    )
    np_linear_background_tw = np_intercept_tw + np_slope_tw_per_year * np_years

    annual_np_detrended_anom_gw = xr.DataArray(
        (np_values_tw - np_linear_background_tw) * TW_TO_GW,
        coords={"year": annual_np_tw["year"].values},
        dims=("year",),
        name="np_wind_work_detrended_anomaly_gw",
    )

    with xr.open_dataset(nino_file, decode_times=True) as ds_nino:
        nino_var = find_data_var(ds_nino, preferred_names=NINO_VAR_CANDIDATES)
        nino_time_name = find_coord_name(ds_nino, ["time", "date"])
        nino_da = ds_nino[nino_var].squeeze().sel(
            {nino_time_name: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")}
        )

        # Niño 3.4 is already an anomaly; use the simple annual mean.
        annual_nino = simple_annual_mean_from_monthly(
            nino_da,
            time_name=nino_time_name,
        ).sel(year=slice(YEAR_START, YEAR_END)).load()

    wind_df = pd.DataFrame({
        "year": annual_np_detrended_anom_gw["year"].values.astype(int),
        "np_wind_work_detrended_anomaly_gw": annual_np_detrended_anom_gw.values.astype(float),
    })
    nino_df = pd.DataFrame({
        "year": annual_nino["year"].values.astype(int),
        "nino34_annual": annual_nino.values.astype(float),
    })

    merged = (
        wind_df.merge(nino_df, on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )

    return {
        "years": merged["year"].to_numpy(dtype=int),
        "np_wind_work_detrended_anomaly_gw": merged["np_wind_work_detrended_anomaly_gw"].to_numpy(dtype=float),
        "nino34_annual": merged["nino34_annual"].to_numpy(dtype=float),
    }


def save_cache(plot_data):
    """Save reusable plotting data."""
    np.savez_compressed(
        CACHE_FILE,
        cache_version=np.array(CACHE_VERSION),
        year_start=np.array(YEAR_START),
        year_end=np.array(YEAR_END),
        lat_min=np.array(LAT_MIN),
        lat_max=np.array(LAT_MAX),
        lon_min=np.array(LON_MIN),
        lon_max=np.array(LON_MAX),
        annual_mean_method=np.array("simple_monthly_mean"),
        anomaly_method=np.array("linear_detrend_residual_1979_2025"),
        **plot_data,
    )


def load_or_compute_plot_data():
    """Load a compatible plotting cache, or rebuild it from source data."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {
                    "years": cache["years"].astype(int),
                    "np_wind_work_detrended_anomaly_gw": cache["np_wind_work_detrended_anomaly_gw"].astype(float),
                    "nino34_annual": cache["nino34_annual"].astype(float),
                }, "cache"

    plot_data = compute_plot_data(WIND_FILE, NINO_FILE)
    save_cache(plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Load plotting data and calculate Pearson correlation
# =============================================================================
plot_data, data_source = load_or_compute_plot_data()

years = plot_data["years"]
np_anom = plot_data["np_wind_work_detrended_anomaly_gw"]
nino = plot_data["nino34_annual"]

valid = np.isfinite(np_anom) & np.isfinite(nino)
if np.count_nonzero(valid) < 3:
    raise ValueError("At least three paired annual values are required for correlation.")

r_value, p_value = stats.pearsonr(np_anom[valid], nino[valid])


# =============================================================================
# 5. Plot Supplementary Figure S3
# =============================================================================
fig, ax_left = plt.subplots(figsize=(7.2, 3.6))
fig.subplots_adjust(left=0.115, right=0.885, bottom=0.16, top=0.965)

# Right axis first so the Niño 3.4 bars remain behind the wind-work series.
ax_right = ax_left.twinx()
bar_colors = np.where(nino >= 0.0, "#D62728", "#1F77B4")
ax_right.bar(
    years,
    nino,
    width=0.72,
    color=bar_colors,
    alpha=0.80,
    edgecolor="none",
    zorder=1,
)

# Keep the left-axis line and markers visually above the bars.
ax_left.set_zorder(2)
ax_right.set_zorder(1)
ax_left.patch.set_visible(False)

ax_left.plot(
    years,
    np_anom,
    color="0.58",
    linewidth=1.2,
    zorder=4,
)
ax_left.scatter(
    years,
    np_anom,
    s=10,
    color="black",
    linewidths=0,
    zorder=5,
)

# Symmetric limits keep zero at the same vertical position on both y axes.
ax_left.set_ylim(symmetric_ylim(np_anom, pad=0.10))
ax_right.set_ylim(symmetric_ylim(nino, pad=0.10))

ax_left.axhline(
    0.0,
    color="#777777",
    linewidth=0.8,
    linestyle="--",
    zorder=3,
)
ax_left.grid(
    True,
    axis="y",
    linestyle="--",
    linewidth=0.7,
    color="#B0B0B0",
    alpha=0.45,
    zorder=2,
)

ax_left.set_xlim(YEAR_START - 1, YEAR_END + 1)
ax_left.set_xticks(np.arange(1980, YEAR_END + 1, 5))
ax_left.set_xlabel("Year")
ax_left.set_ylabel("Detrended North Pacific wind-work anomaly (GW)")
ax_right.set_ylabel(r"Niño 3.4 anomaly (°C)")

ax_left.tick_params(axis="both", width=0.8, length=3)
ax_right.tick_params(axis="y", width=0.8, length=3)

# Correlation annotation; the figure title is provided in the manuscript legend.
ax_left.text(
    0.015,
    0.965,
    rf"$r={r_value:.2f}$, {format_p(p_value)}",
    transform=ax_left.transAxes,
    ha="left",
    va="top",
    fontsize=7.2,
)

legend_handles = [
    Line2D(
        [0], [0],
        color="0.58",
        linewidth=1.2,
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=3.2,
        label="Detrended North Pacific wind-work anomaly",
    ),
    Patch(
        facecolor="#D62728",
        edgecolor="none",
        alpha=0.80,
        label="Niño 3.4 anomaly (+)",
    ),
    Patch(
        facecolor="#1F77B4",
        edgecolor="none",
        alpha=0.80,
        label="Niño 3.4 anomaly (−)",
    ),
]
ax_left.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=False,
    handlelength=1.8,
    borderaxespad=0.35,
    labelspacing=0.35,
)


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
print(f"Pearson correlation: r={r_value:.3f}, P={p_value:.6g}")
print(f"Saved figure: {FIGURE_FILE}")
print(f"Plotting cache: {CACHE_FILE}")
