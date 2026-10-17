# -*- coding: utf-8 -*-
"""Generate Figure 1 from source data or a reusable plotting-data cache.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import numpy as np
import xarray as xr
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

WIND_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
CACHE_FILE = CACHE_DIR / "Fig1_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "Figure1.png"

POWER_VAR = "power_input"
YEAR_START = 1979
YEAR_END = 2025
HAC_LAG = 3
CACHE_VERSION = 2
RECOMPUTE_CACHE = False

R_EARTH = 6_371_000.0

# North Pacific analysis domain: 0–60°N, 110°E–100°W.
D1_REGION = {
    "lat_min": 0.0,
    "lat_max": 60.0,
    "lon_min": 110.0,
    "lon_max": 260.0,  # 100°W = 260°E
}

HIGHLIGHT_YEARS = np.array([1997, 1998, 2015, 2016], dtype=int)


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
    """Return the first matching coordinate or dimension name."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def convert_longitude_to_360(ds, lon_name):
    """Convert longitude coordinates to 0–360° and sort them."""
    lon360 = xr.where(ds[lon_name] < 0, ds[lon_name] + 360, ds[lon_name])
    return ds.assign_coords({lon_name: lon360}).sortby(lon_name)


def compute_grid_area(ds, lat_name, lon_name):
    """Compute regular-grid spherical cell area as a latitude-dependent array."""
    lat = ds[lat_name]
    lon = ds[lon_name]

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Latitude and longitude coordinates must each contain at least two points.")

    d_lat = float(np.abs(lat.values[1] - lat.values[0]))
    d_lon = float(np.abs(lon.values[1] - lon.values[0]))

    return (
        R_EARTH**2
        * np.deg2rad(d_lat)
        * np.deg2rad(d_lon)
        * np.cos(np.deg2rad(lat))
    )


def annual_mean_series(monthly_da, time_name, output_name):
    """Convert a monthly time series to annual means for the study period."""
    import pandas as pd

    annual_da = (
        monthly_da
        .resample({time_name: "1YE"})
        .mean(dim=time_name, skipna=True)
        .load()
    )

    out = pd.DataFrame({
        "year": annual_da[time_name].dt.year.values.astype(int),
        output_name: annual_da.values.astype(float),
    })

    return (
        out.loc[
            (out["year"] >= YEAR_START) & (out["year"] <= YEAR_END)
        ]
        .sort_values("year")
        .reset_index(drop=True)
    )


def calc_trend_and_hac(years, values, hac_lag=HAC_LAG):
    """Fit an OLS linear trend and compute its HAC-robust P value."""
    import statsmodels.api as sm

    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 4:
        raise ValueError("At least four annual values are required for trend estimation.")

    xbar = float(np.mean(x))
    xc = x - xbar

    design = sm.add_constant(xc)
    ols = sm.OLS(y, design).fit()
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=min(hac_lag, x.size - 1),
    )

    xfit = np.linspace(x.min(), x.max(), 300)
    yfit = ols.params[0] + ols.params[1] * (xfit - xbar)

    return {
        "slope_TW_per_year": float(ols.params[1]),
        "trend_GW_per_dec": float(ols.params[1] * 10_000.0),
        "intercept_centered": float(ols.params[0]),
        "xbar": xbar,
        "p": float(hac.pvalues[1]),
        "xfit": xfit,
        "yfit": yfit,
    }


def format_p(p_value):
    """Format P values for figure annotations."""
    if p_value < 0.001:
        return r"$P<0.001$"
    return rf"$P={p_value:.3f}$"


def draw_d1_box(ax, region):
    """Draw the North Pacific analysis domain, splitting edges at the dateline."""
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]

    line_kwargs = {
        "transform": ccrs.PlateCarree(),
        "color": "#56B4E9",
        "linestyle": "--",
        "linewidth": 1.0,
        "alpha": 0.75,
        "zorder": 8,
    }

    lon_max_plot = lon_max if lon_max <= 180.0 else lon_max - 360.0

    ax.plot([lon_min, lon_min], [lat_min, lat_max], **line_kwargs)
    ax.plot([lon_max_plot, lon_max_plot], [lat_min, lat_max], **line_kwargs)

    east = np.linspace(lon_min, 180.0, 150)
    west = np.linspace(-180.0, lon_max - 360.0, 150)

    for lat_value in (lat_min, lat_max):
        ax.plot(east, np.full_like(east, lat_value), **line_kwargs)
        ax.plot(west, np.full_like(west, lat_value), **line_kwargs)



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
    """Check whether an existing cache matches the current Figure 1 settings."""
    required_keys = {
        "cache_version",
        "year_start",
        "year_end",
        "hac_lag",
        "d1_region",
        "latitude",
        "longitude_cyclic",
        "mean_cyclic",
        "mean_color_limit",
        "years",
        "global_tw",
        "d1_tw",
        "d1_linear_background_tw",
        "d1_detrended_residual_tw",
        "global_background_adjusted_tw",
        "original_xfit",
        "original_yfit",
        "original_trend_GW_per_dec",
        "original_p",
        "adjusted_xfit",
        "adjusted_yfit",
        "adjusted_trend_GW_per_dec",
        "adjusted_p",
    }

    if not required_keys.issubset(cache.files):
        return False

    region_now = np.array([
        D1_REGION["lat_min"],
        D1_REGION["lat_max"],
        D1_REGION["lon_min"],
        D1_REGION["lon_max"],
    ], dtype=float)

    return (
        int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and int(cache["hac_lag"]) == HAC_LAG
        and np.allclose(cache["d1_region"], region_now)
    )


def compute_plot_data(source_file):
    """Compute all data required to reproduce Figure 1."""
    with xr.open_dataset(source_file, decode_times=True) as ds:
        time_name = find_coord_name(ds, ["time", "Time", "TIME"])
        lat_name = find_coord_name(ds, ["latitude", "lat", "Latitude", "LAT"])
        lon_name = find_coord_name(ds, ["longitude", "lon", "Longitude", "LON"])

        if time_name is None or lat_name is None or lon_name is None:
            raise KeyError("Could not identify time, latitude, and longitude coordinates.")
        if POWER_VAR not in ds:
            raise KeyError(f"Variable '{POWER_VAR}' was not found in {source_file.name}.")

        ds = ds.sel({
            time_name: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")
        })
        ds = convert_longitude_to_360(ds, lon_name)
        ds = ds.sortby(lat_name)

        wind_da = ds[POWER_VAR].transpose(time_name, lat_name, lon_name)
        latitude = ds[lat_name].values.astype(float)
        longitude = ds[lon_name].values.astype(float)

        # Panel a: time-mean wind work, displayed only from 3° to 60° latitude.
        mean_map = wind_da.mean(dim=time_name, skipna=True) * 1000.0
        lat_mask_plot = (
            (np.abs(ds[lat_name]) >= 3.0)
            & (np.abs(ds[lat_name]) <= 60.0)
        )
        mean_map = mean_map.where(lat_mask_plot)

        mean_cyclic, longitude_cyclic = add_cyclic_point(
            mean_map.values,
            coord=longitude,
        )
        mean_color_limit = float(np.nanpercentile(mean_map.values, 99))

        # Area-integrated monthly wind work.
        grid_area_by_lat = compute_grid_area(ds, lat_name, lon_name)

        global_monthly_tw = (
            (wind_da * grid_area_by_lat)
            .sum(dim=[lat_name, lon_name], skipna=True)
            / 1e12
        )

        d1_mask = (
            (ds[lat_name] >= D1_REGION["lat_min"])
            & (ds[lat_name] <= D1_REGION["lat_max"])
            & (ds[lon_name] >= D1_REGION["lon_min"])
            & (ds[lon_name] <= D1_REGION["lon_max"])
        )

        d1_monthly_tw = (
            (wind_da.where(d1_mask) * grid_area_by_lat)
            .sum(dim=[lat_name, lon_name], skipna=True)
            / 1e12
        )

        df_global = annual_mean_series(global_monthly_tw, time_name, "global_TW")
        df_d1 = annual_mean_series(d1_monthly_tw, time_name, "d1_TW")
        df = df_global.merge(df_d1, on="year", how="inner")

    # Remove the detrended D1 residual from the global series while retaining
    # the D1 long-term linear background.
    fit_d1 = calc_trend_and_hac(df["year"], df["d1_TW"])
    d1_linear_background = (
        fit_d1["intercept_centered"]
        + fit_d1["slope_TW_per_year"]
        * (df["year"].to_numpy(dtype=float) - fit_d1["xbar"])
    )

    d1_detrended_residual = df["d1_TW"].to_numpy(dtype=float) - d1_linear_background
    global_background_adjusted = (
        df["global_TW"].to_numpy(dtype=float) - d1_detrended_residual
    )

    fit_original = calc_trend_and_hac(df["year"], df["global_TW"])
    fit_adjusted = calc_trend_and_hac(df["year"], global_background_adjusted)

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "hac_lag": np.array(HAC_LAG, dtype=np.int16),
        "d1_region": np.array([
            D1_REGION["lat_min"],
            D1_REGION["lat_max"],
            D1_REGION["lon_min"],
            D1_REGION["lon_max"],
        ], dtype=float),
        "latitude": latitude,
        "longitude_cyclic": longitude_cyclic,
        "mean_cyclic": mean_cyclic,
        "mean_color_limit": np.array(mean_color_limit, dtype=float),
        "years": df["year"].to_numpy(dtype=int),
        "global_tw": df["global_TW"].to_numpy(dtype=float),
        "d1_tw": df["d1_TW"].to_numpy(dtype=float),
        "d1_linear_background_tw": np.asarray(d1_linear_background, dtype=float),
        "d1_detrended_residual_tw": np.asarray(d1_detrended_residual, dtype=float),
        "global_background_adjusted_tw": np.asarray(global_background_adjusted, dtype=float),
        "original_xfit": fit_original["xfit"],
        "original_yfit": fit_original["yfit"],
        "original_trend_GW_per_dec": np.array(fit_original["trend_GW_per_dec"], dtype=float),
        "original_p": np.array(fit_original["p"], dtype=float),
        "adjusted_xfit": fit_adjusted["xfit"],
        "adjusted_yfit": fit_adjusted["yfit"],
        "adjusted_trend_GW_per_dec": np.array(fit_adjusted["trend_GW_per_dec"], dtype=float),
        "adjusted_p": np.array(fit_adjusted["p"], dtype=float),
    }


def load_or_build_plot_data():
    """Load a compatible cache or rebuild it from the source NetCDF file."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    if not WIND_FILE.exists():
        raise FileNotFoundError(
            "Figure 1 cannot be generated because neither a compatible plotting "
            "cache nor the source NetCDF file is available.\n\n"
            f"Expected plotting cache:\n  {CACHE_FILE}\n\n"
            f"Expected source data for the first run:\n  {WIND_FILE}\n\n"
            "Place the source NetCDF file in the project 'data' folder for the "
            "first run, or place a compatible Fig1_plot_data.npz file in 'cache'."
        )

    plot_data = compute_plot_data(WIND_FILE)
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Load or compute plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()

latitude = data["latitude"]
longitude_cyclic = data["longitude_cyclic"]
mean_cyclic = data["mean_cyclic"]
mean_color_limit = float(data["mean_color_limit"])

years = data["years"].astype(int)
global_tw = data["global_tw"].astype(float)
global_adjusted_tw = data["global_background_adjusted_tw"].astype(float)

original_xfit = data["original_xfit"].astype(float)
original_yfit = data["original_yfit"].astype(float)
adjusted_xfit = data["adjusted_xfit"].astype(float)
adjusted_yfit = data["adjusted_yfit"].astype(float)
adjusted_trend = float(data["adjusted_trend_GW_per_dec"])
adjusted_p = float(data["adjusted_p"])


# =============================================================================
# 5. Figure 1
# =============================================================================
# Width ≈ 183 mm, suitable for a full-width journal figure.
fig = plt.figure(figsize=(7.2, 8.0), facecolor="white")

pos_map = [0.08, 0.56, 0.84, 0.37]
pos_cbar = [0.26, 0.49, 0.45, 0.014]
pos_ts = [0.11, 0.095, 0.81, 0.305]

# Panel a: mean wind-work map
ax_map = fig.add_axes(
    pos_map,
    projection=ccrs.Robinson(central_longitude=180),
)

map_image = ax_map.pcolormesh(
    longitude_cyclic,
    latitude,
    mean_cyclic,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    vmin=0.0,
    vmax=mean_color_limit,
    shading="auto",
    rasterized=True,
)

ax_map.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=2)
ax_map.coastlines(linewidth=0.5, zorder=3)
ax_map.set_global()
ax_map.set_title("a", loc="left", fontweight="bold", pad=4)

# Draw gridlines without automatic labels. Latitude labels are added manually
# on the left only, which avoids duplicate labels on Robinson-map boundaries.
gridlines = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.7,
    color="#666666",
    alpha=0.45,
    linestyle="--",
)
gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
gridlines.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

# Separate invisible gridliner used only for bottom longitude labels.
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
draw_d1_box(ax_map, D1_REGION)

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

# Panel b: annual global wind work
ax_ts = fig.add_axes(pos_ts)

ax_ts.plot(
    years,
    global_tw,
    color="#B0B0B0",
    linewidth=1.2,
    alpha=0.95,
    zorder=1,
)
ax_ts.scatter(
    years,
    global_tw,
    color="#7F7F7F",
    s=15,
    alpha=0.85,
    label="Original global wind work",
    zorder=2,
)

ax_ts.plot(
    years,
    global_adjusted_tw,
    color="#4D4D4D",
    linewidth=1.2,
    alpha=0.95,
    zorder=3,
)
ax_ts.scatter(
    years,
    global_adjusted_tw,
    color="#000000",
    s=17,
    alpha=0.90,
    label="North Pacific background-adjusted global wind work",
    zorder=4,
)

highlight_mask = np.isin(years, HIGHLIGHT_YEARS)
ax_ts.scatter(
    years[highlight_mask],
    global_tw[highlight_mask],
    color="#E41A1C",
    edgecolor="black",
    linewidth=0.5,
    s=26,
    zorder=8,
)

ax_ts.plot(
    original_xfit,
    original_yfit,
    color="#7F7F7F",
    linestyle="--",
    linewidth=1.4,
    alpha=0.95,
    zorder=5,
)

adjusted_label = (
    rf"1979–2025 trend: {adjusted_trend:+.1f} GW decade$^{{-1}}$, "
    + format_p(adjusted_p)
)
ax_ts.plot(
    adjusted_xfit,
    adjusted_yfit,
    color="#E41A1C",
    linestyle="--",
    linewidth=1.4,
    alpha=0.95,
    label=adjusted_label,
    zorder=6,
)

ax_ts.set_ylabel("Global wind work (TW)")
ax_ts.set_xlabel("Year")
ax_ts.set_title("b", loc="left", fontweight="bold", pad=4)

ax_ts.set_xlim(YEAR_START - 1, YEAR_END + 1)
ax_ts.set_xticks(np.arange(1980, 2026, 5))

all_values = np.concatenate([global_tw, global_adjusted_tw])
y_min = float(np.nanmin(all_values))
y_max = float(np.nanmax(all_values))
y_range = y_max - y_min
if y_range <= 0:
    y_range = max(abs(y_max), 1.0) * 0.1

ax_ts.set_ylim(
    y_min - 0.15 * y_range,
    y_max + 0.22 * y_range,
)
ax_ts.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    color="#B0B0B0",
    alpha=0.45,
)
ax_ts.legend(
    loc="lower right",
    bbox_to_anchor=(0.995, 0.03),
    frameon=False,
    handlelength=2.0,
    borderaxespad=0.25,
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
