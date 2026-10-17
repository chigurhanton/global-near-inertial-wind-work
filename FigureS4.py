# -*- coding: utf-8 -*-
"""Climatological mean mixed-layer depth (1979-2025) -- panel b only.

This is a single-panel extraction of panel b from Figure_climatology.py.

The map style is kept IDENTICAL to the original figure:
    * Robinson projection, central longitude 180, global
    * ocean-only, 3-60 degrees latitude (both hemispheres)
    * grey land, 110m coastlines, dashed graticule
    * latitude labels on the left map boundary, longitude labels at the bottom
    * bold lowercase panel letter, top-left
    * slim horizontal colorbar with the unit written to its right

The ONLY change is the colormap (see the CMAP constant below).

Run from the PROJECT ROOT:
    python Figure_climatology_panel_b.py

Inputs:
    cache/Fig_climatology_plot_data.npz
        Shared with Figure_climatology.py. Already contains mld_mean, so the
        multi-gigabyte source files are normally NOT read.
    data/global_wind_stress_ni_total_variance_1979_2025.nc
        Used only to recover the analysis grid if the cache is missing.
    data/mld_all_1979to2025.mat
        Used only if the cache is missing.

Outputs:
    plots/Figure_climatology_panel_b.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

try:
    import h5py
except Exception:
    h5py = None

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


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

NC_STRESS_PATH = DATA_DIR / "global_wind_stress_ni_total_variance_1979_2025.nc"
MLD_FILE = DATA_DIR / "mld_all_1979to2025.mat"

# Reuse the cache written by Figure_climatology.py (read-only here).
CACHE_FILE = CACHE_DIR / "FigS4_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS4.png"

NI_VARIABLE = "resonant_ni_wind_stress_variance"
MLD_VAR = "mld_all"

START_YEAR = 1979
END_YEAR = 2025
YEAR_SLICE = slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")

MIN_ABS_LAT = 3.0
MAX_ABS_LAT = 60.0

CACHE_VERSION = 1
RECOMPUTE_CACHE = False

# Colour limits are taken from percentiles of the mean field so that the
# scaling adapts if the source product changes.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

# Panel letter. Empty by default: this is a standalone single-panel figure,
# so a "b" label would be meaningless. Set to e.g. "b" if the figure is later
# placed next to a panel a.
PANEL_LABEL = ""

FIG_WIDTH = 5.4
FIG_HEIGHT = 3.2


# =============================================================================
# 2. Colormap
# =============================================================================
# Panel b shows mixed-layer depth in metres; larger values = deeper mixed layer.
#
# "YlGnBu" reads naturally for a depth field: pale yellow (shallow) through
# teal to deep blue (deep). It is also clearly distinct from the viridis used
# in the two-panel original, so the two figures do not look interchangeable.
#
# Swap the string below for any other matplotlib colormap. Options that work
# well for a depth field:
#
#   "YlGnBu"      pale yellow -> deep blue        (default)
#   "GnBu"        pale -> blue, greener mid-range
#   "PuBu"        pale -> blue, softer / more muted
#   "Spectral_r"  blue -> red, high contrast, very legible
#   "RdYlBu_r"    blue -> red, muted and paper-friendly
#   "turbo"       vivid rainbow, maximum contrast
#   "cividis"     colour-blind safe, darker overall
#
# Reversible variants (append "_r") flip the direction, so "_r" makes shallow
# water dark instead of pale.
CMAP = "YlGnBu"


# =============================================================================
# 3. Figure style (identical to Figure 3 / Figure_climatology.py)
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
# 4. Utilities
# =============================================================================
def rename_latlon(da):
    """Rename common latitude/longitude names to lat/lon."""
    mapping = {}
    if "latitude" in da.coords or "latitude" in da.dims:
        mapping["latitude"] = "lat"
    if "longitude" in da.coords or "longitude" in da.dims:
        mapping["longitude"] = "lon"
    if mapping:
        da = da.rename(mapping)
    return da


def apply_analysis_mask(da):
    """Keep 3-60 degrees latitude in both hemispheres."""
    lat_abs = np.abs(da["lat"])
    return da.where((lat_abs >= MIN_ABS_LAT) & (lat_abs <= MAX_ABS_LAT))


def read_mld_as_time_lat_lon(path):
    """Read mld_all from MAT/HDF5 and return DataArray(time, lat, lon)."""
    if h5py is not None:
        try:
            with h5py.File(path, "r") as handle:
                if MLD_VAR not in handle:
                    raise KeyError(f"Variable '{MLD_VAR}' not found. Keys: {list(handle.keys())}")
                data = handle[MLD_VAR][()]
        except OSError:
            if loadmat is None:
                raise
            mat = loadmat(path)
            if MLD_VAR not in mat:
                available = [k for k in mat if not k.startswith("__")]
                raise KeyError(f"Variable '{MLD_VAR}' not found. Variables: {available}")
            data = mat[MLD_VAR]
    else:
        if loadmat is None:
            raise ImportError("Neither h5py nor scipy.io.loadmat is available.")
        mat = loadmat(path)
        if MLD_VAR not in mat:
            available = [k for k in mat if not k.startswith("__")]
            raise KeyError(f"Variable '{MLD_VAR}' not found. Variables: {available}")
        data = mat[MLD_VAR]

    data = np.asarray(data, dtype=float).squeeze()
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D MLD array; obtained shape={data.shape}")

    data[~np.isfinite(data)] = np.nan
    data[np.abs(data) > 1.0e4] = np.nan

    expected_months = (END_YEAR - START_YEAR + 1) * 12
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

    time = pd.date_range(start=f"{START_YEAR}-01-01", periods=ntime, freq="MS")
    lat = np.linspace(-80.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)

    da = xr.DataArray(
        data_tyl,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="mld",
        attrs={"units": "m"},
    )
    da = da.sel(time=YEAR_SLICE).sortby("lat").sortby("lon")
    return da


def reconstruct_da(array, lat, lon, name):
    """Reconstruct a DataArray from cached arrays."""
    return xr.DataArray(array, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)


def percentile_limits(da, low_percentile, high_percentile):
    """Colour limits from percentiles of the finite values."""
    values = np.asarray(da.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(values, low_percentile))
    vmax = float(np.nanpercentile(values, high_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def area_weighted_mean(da):
    """Cosine-latitude weighted spatial mean over valid points."""
    weights = np.cos(np.deg2rad(da["lat"])).broadcast_like(da).where(np.isfinite(da))
    return float(((da * weights).sum() / weights.sum()).values)


def add_left_latitude_labels(ax, latitudes=(-60, -30, 0, 30, 60), fontsize=7.2):
    """Place latitude labels only along the left Robinson-map boundary."""
    latitudes = np.asarray(latitudes, dtype=float)
    seam_lon = np.zeros_like(latitudes)
    points = ax.projection.transform_points(ccrs.PlateCarree(), seam_lon, latitudes)

    x_min, x_max = ax.get_xlim()
    x_offset = 0.012 * (x_max - x_min)

    for lat_value, (x_proj, y_proj, _) in zip(latitudes, points):
        if lat_value < 0:
            label = f"{int(abs(lat_value))}\u00b0S"
        elif lat_value > 0:
            label = f"{int(lat_value)}\u00b0N"
        else:
            label = "0\u00b0"
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
    """Draw dashed gridlines and optional bottom longitude labels."""
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


def plot_panel(
    ax,
    cbar_ax,
    da,
    panel_label,
    cmap,
    unit_text=None,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    cbar_formatter=None,
    show_bottom=False,
):
    """Plot one Robinson map of a climatological mean field with a compact colorbar."""
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
        zorder=1,
        rasterized=True,
    )

    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=4)
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=8)
    ax.set_global()
    add_map_gridlines(ax, show_bottom)
    if panel_label:
        ax.set_title(panel_label, loc="left", fontweight="bold", pad=4)

    colorbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal", ticks=cbar_ticks)
    colorbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
    colorbar.outline.set_linewidth(0.8)
    if cbar_formatter is not None:
        colorbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(cbar_formatter))
        colorbar.ax.minorticks_off()

    if unit_text is not None and unit_text != "":
        colorbar.ax.text(
            1.03,
            0.5,
            unit_text,
            transform=colorbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
        )
    return image


# =============================================================================
# 5. Cache
# =============================================================================
def cache_is_compatible(cache):
    required = {
        "cache_version",
        "start_year",
        "end_year",
        "lat",
        "lon",
        "mld_mean",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["start_year"]) == START_YEAR
        and int(cache["end_year"]) == END_YEAR
    )


def compute_plot_data():
    """Compute the climatological mean mixed-layer depth from source data."""
    required_files = [NC_STRESS_PATH, MLD_FILE]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Panel b cannot be generated because required source "
            "files are missing:\n\n" + "\n".join(str(path) for path in missing)
        )

    # The analysis grid is defined by the near-inertial wind-stress product,
    # exactly as in Figure_climatology.py.
    with xr.open_dataset(NC_STRESS_PATH) as ds_stress:
        reference_grid = (
            rename_latlon(ds_stress[NI_VARIABLE])
            .isel(time=0, drop=True)
            .sortby("lat")
            .sortby("lon")
        )

    mld = read_mld_as_time_lat_lon(MLD_FILE)
    mld = mld.interp_like(reference_grid, method="nearest")
    mld = apply_analysis_mask(mld)
    mld_mean = mld.mean(dim="time", skipna=True)

    plot_data = {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "start_year": np.array(START_YEAR, dtype=np.int16),
        "end_year": np.array(END_YEAR, dtype=np.int16),
        "lat": mld_mean["lat"].values,
        "lon": mld_mean["lon"].values,
        "mld_mean": np.asarray(mld_mean.values),
    }
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


def load_or_build_plot_data():
    """Load compatible cache or rebuild from source data."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    return compute_plot_data()


# =============================================================================
# 6. Load plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()
lat = data["lat"]
lon = data["lon"]

mld_mean = reconstruct_da(data["mld_mean"], lat, lon, "mld_mean")


# =============================================================================
# 7. Colour limits
# =============================================================================
mld_vmin, mld_vmax = percentile_limits(mld_mean, LOW_PERCENTILE, HIGH_PERCENTILE)

print(f"Panel b colour range: {mld_vmin:.3g} to {mld_vmax:.3g} m "
      f"(linear scale, cmap = {CMAP})")


# =============================================================================
# 8. Figure
# =============================================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
projection = ccrs.Robinson(central_longitude=180)

# Single map, same axes width as the original two-panel layout, with the
# colorbar sitting directly underneath in the same slim style.
map_position = [0.08, 0.235, 0.84, 0.69]
cbar_position = [0.29, 0.115, 0.42, 0.024]

ax = fig.add_axes(map_position, projection=projection)
cbar_ax = fig.add_axes(cbar_position)

plot_panel(
    ax=ax,
    cbar_ax=cbar_ax,
    da=mld_mean,
    panel_label=PANEL_LABEL,
    cmap=CMAP,
    unit_text=r"(m)",
    vmin=mld_vmin,
    vmax=mld_vmax,
    show_bottom=True,
)

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

print(f"\nPlot data: {data_source}")
print(f"Saved figure: {FIGURE_FILE}")
print(f"Plotting cache: {CACHE_FILE}")


# =============================================================================
# 9. Diagnostics
# =============================================================================
print("\n" + "=" * 68)
print("CLIMATOLOGICAL MEAN MIXED-LAYER DEPTH 1979-2025 (ocean, 3-60 deg)")
print("=" * 68)

mld_values = mld_mean.values[np.isfinite(mld_mean.values)]

print(f"   area-weighted mean : {area_weighted_mean(mld_mean):.2f} m")
print(f"   median             : {np.median(mld_values):.2f} m")
print(f"   p10 / p90          : {np.percentile(mld_values, 10):.2f} / "
      f"{np.percentile(mld_values, 90):.2f} m")
print(f"   min / max          : {mld_values.min():.2f} / {mld_values.max():.2f} m")
print("=" * 68)
