# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S5 from source data or a reusable cache.

The figure compares global near-inertial-wave speed biases relative to GDP
surface drifters for the slab-model estimate and the DL-corrected estimate.
Designed for execution as a single cell in a VS Code Jupyter notebook.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import binned_statistic_2d


# =============================================================================
# 1. Project paths and analysis settings
# =============================================================================
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "cache"
PLOTS_DIR = PROJECT_DIR / "plots"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFTER_FILE = DATA_DIR / "gdp_drifter_NIW_300h_segments.mat"
SLAB_FILE = DATA_DIR / "3D_mean_inertial_KE_1979_2025_final.nc"
DL_FILE = DATA_DIR / "niw_monthly_1979to2025_with_oras5mld.zarr"

CACHE_FILE = CACHE_DIR / "FigS6_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS6.png"

YEAR_START = 1979
YEAR_END = 2025
GRID_RES = 2.0
MAX_BIAS = 5.0
CMAP_BIAS = "RdBu_r"
CACHE_VERSION = 1
RECOMPUTE_CACHE = False


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
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
})


# =============================================================================
# 3. Data helpers
# =============================================================================
def robust_read_mat(filepath, variable_names):
    """Read selected variables from either classic or HDF5-backed MAT files."""
    try:
        import scipy.io as sio

        mat_data = sio.loadmat(filepath)
        return {name: mat_data[name] for name in variable_names}
    except (NotImplementedError, OSError, ValueError):
        import h5py

        with h5py.File(filepath, "r") as handle:
            return {name: handle[name][()] for name in variable_names}


def cache_is_compatible(cache):
    """Check whether an existing plotting cache matches the current settings."""
    required = {
        "cache_version",
        "year_start",
        "year_end",
        "grid_res",
        "lon_grid",
        "lat_grid",
        "bias_slab",
        "bias_dl",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and np.isclose(float(cache["grid_res"]), GRID_RES)
    )


def build_plot_data():
    """Compute the two 2-degree bias fields used by Supplementary Figure S5."""
    missing = [path for path in (DRIFTER_FILE, SLAB_FILE, DL_FILE) if not path.exists()]
    if missing:
        paths = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Supplementary Figure S5 source data are missing:\n"
            f"{paths}\n\n"
            "Place all required source files in the project 'data' folder."
        )

    # GDP drifter reference field.
    drifter_data = robust_read_mat(
        DRIFTER_FILE,
        ["uI_300", "vI_300", "lon_300", "lat_300"],
    )
    u_all = np.asarray(drifter_data["uI_300"], dtype=float)
    v_all = np.asarray(drifter_data["vI_300"], dtype=float)
    lon_all = np.asarray(drifter_data["lon_300"], dtype=float)
    lat_all = np.asarray(drifter_data["lat_300"], dtype=float)

    time_axis = 0 if u_all.shape[0] == 300 else 1
    lon_pts = lon_all[0, :] if time_axis == 0 else lon_all[:, 0]
    lat_pts = lat_all[0, :] if time_axis == 0 else lat_all[:, 0]
    lon_pts = (lon_pts + 180.0) % 360.0 - 180.0

    speed_drifter = np.nanmean(
        np.sqrt(u_all**2 + v_all**2) * 100.0,
        axis=time_axis,
    )

    lon_bins = np.arange(-180.0, 180.0 + GRID_RES, GRID_RES)
    lat_bins = np.arange(-90.0, 90.0 + GRID_RES, GRID_RES)
    drifter_grid, lon_edges, lat_edges, _ = binned_statistic_2d(
        lon_pts,
        lat_pts,
        speed_drifter,
        statistic="mean",
        bins=[lon_bins, lat_bins],
    )
    drifter_grid = drifter_grid.T

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Slab-model estimate.
    with xr.open_dataset(SLAB_FILE) as ds_slab:
        if "inertial_KE" not in ds_slab:
            raise KeyError(f"Variable 'inertial_KE' was not found in {SLAB_FILE.name}.")
        if "longitude" not in ds_slab.coords or "latitude" not in ds_slab.coords:
            raise KeyError("The slab-model dataset must contain latitude/longitude coordinates.")

        ds_slab = ds_slab.assign_coords(
            longitude=((ds_slab.longitude + 180.0) % 360.0) - 180.0
        ).sortby("longitude")
        ke_mean = ds_slab["inertial_KE"].mean(dim="time", skipna=True).compute()
        speed_slab_mean = np.sqrt(2.0 * ke_mean) * 100.0
        speed_slab_grid = speed_slab_mean.interp(
            latitude=lat_centers,
            longitude=lon_centers,
        ).values

    bias_slab = speed_slab_grid - drifter_grid

    # DL-corrected estimate.
    ds_dl = xr.open_zarr(DL_FILE, consolidated=False)
    try:
        if "speed" not in ds_dl:
            raise KeyError(f"Variable 'speed' was not found in {DL_FILE.name}.")

        speed_dl = ds_dl["speed"].transpose("time", "lat", "lon")
        speed_dl = speed_dl.where(
            (speed_dl["time.year"] >= YEAR_START)
            & (speed_dl["time.year"] <= YEAR_END),
            drop=True,
        )
        speed_dl_mean = (speed_dl * 100.0).mean("time", skipna=True)
        speed_dl_mean = speed_dl_mean.assign_coords(
            lon=((speed_dl_mean.lon + 180.0) % 360.0) - 180.0
        ).sortby("lon").compute()
        speed_dl_grid = speed_dl_mean.interp(
            lat=lat_centers,
            lon=lon_centers,
        ).values
    finally:
        ds_dl.close()

    bias_dl = speed_dl_grid - drifter_grid

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "grid_res": np.array(GRID_RES, dtype=np.float32),
        "lon_grid": lon_grid.astype(np.float32),
        "lat_grid": lat_grid.astype(np.float32),
        "bias_slab": bias_slab.astype(np.float32),
        "bias_dl": bias_dl.astype(np.float32),
    }


def load_or_build_plot_data():
    """Load a compatible plotting cache or rebuild it from the source datasets."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    plot_data = build_plot_data()
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Map helpers
# =============================================================================
def add_left_latitude_labels(ax, latitudes=(-60, -30, 0, 30, 60), fontsize=7.2):
    """Place latitude labels only along the left Robinson-map boundary."""
    latitudes = np.asarray(latitudes, dtype=float)

    # For Robinson(central_longitude=180), longitude 0° lies on the left seam.
    # Using projected seam coordinates makes each label follow the curved map
    # boundary instead of forming a detached vertical column.
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


def add_standard_gridlines(ax):
    """Draw gridlines above the bias field without automatic latitude labels."""
    # Longitude gridlines and bottom longitude labels only.
    gl_lon = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.45,
        linestyle="--",
        zorder=2.2,
    )
    gl_lon.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl_lon.ylocator = mticker.FixedLocator([])
    gl_lon.top_labels = False
    gl_lon.bottom_labels = True
    gl_lon.left_labels = False
    gl_lon.right_labels = False
    if hasattr(gl_lon, "geo_labels"):
        gl_lon.geo_labels = False
    gl_lon.xlabel_style = {"size": 7.2, "color": "#333333"}
    try:
        gl_lon.x_inline = False
        gl_lon.y_inline = False
    except Exception:
        pass

    # Latitude gridlines only; all latitude labels are added manually.
    gl_lat = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.6,
        color="gray",
        alpha=0.45,
        linestyle="--",
        zorder=2.2,
    )
    gl_lat.xlocator = mticker.FixedLocator([])
    gl_lat.ylocator = mticker.FixedLocator(np.arange(-60, 61, 30))

    add_left_latitude_labels(ax)


def plot_bias_map(ax, lon_grid, lat_grid, bias, panel_label):
    """Plot one Robinson-projection NIW-speed bias map."""
    image = ax.pcolormesh(
        lon_grid,
        lat_grid,
        bias,
        transform=ccrs.PlateCarree(),
        cmap=CMAP_BIAS,
        vmin=-MAX_BIAS,
        vmax=MAX_BIAS,
        shading="auto",
        zorder=1,
    )

    ax.set_global()
    add_standard_gridlines(ax)
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=3)
    ax.coastlines(linewidth=0.5, zorder=4)
    ax.set_title(panel_label, loc="left", fontweight="bold", pad=3)
    return image


# =============================================================================
# 5. Load plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()
lon_grid = data["lon_grid"].astype(float)
lat_grid = data["lat_grid"].astype(float)
bias_slab = data["bias_slab"].astype(float)
bias_dl = data["bias_dl"].astype(float)


# =============================================================================
# 6. Supplementary Figure S5
# =============================================================================
fig = plt.figure(figsize=(6.3, 5.25), facecolor="white")
projection = ccrs.Robinson(central_longitude=180)

# Compact two-map layout. Cartopy preserves the Robinson aspect ratio inside
# these axes, so the shared colorbar is positioned from the maps' final drawn
# right edge rather than from the nominal axes rectangle.
pos_a = [0.070, 0.555, 0.790, 0.350]
pos_b = [0.070, 0.145, 0.790, 0.350]

ax_a = fig.add_axes(pos_a, projection=projection)
ax_b = fig.add_axes(pos_b, projection=projection)

image = plot_bias_map(ax_a, lon_grid, lat_grid, bias_slab, "a")
plot_bias_map(ax_b, lon_grid, lat_grid, bias_dl, "b")

# Shared vertical colorbar on the right. Obtain the active GeoAxes positions
# after Cartopy has applied the Robinson aspect ratio, then keep only a small
# gap between the map boundary and the colorbar.
fig.canvas.draw()
map_right = max(ax_a.get_position().x1, ax_b.get_position().x1)
cbar_gap = 0.040
cbar_height = 0.235
cbar_bottom = 0.525 - cbar_height / 2.0
cbar_width = 0.019
cax = fig.add_axes([map_right + cbar_gap, cbar_bottom, cbar_width, cbar_height])
cbar = fig.colorbar(image, cax=cax, orientation="vertical")
cbar.ax.yaxis.set_ticks_position("right")
cbar.ax.yaxis.set_label_position("right")
cbar.ax.tick_params(axis="y", labelsize=7.2, width=0.7, length=2.5)
cbar.outline.set_linewidth(0.7)

# Horizontal unit label at the top of the vertical colorbar.
cbar.ax.text(
    0.5,
    1.055,
    r"(cm s$^{-1}$)",
    transform=cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.0,
)


# =============================================================================
# 7. Save as a 400-dpi RGB PNG
# =============================================================================
fig.savefig(
    FIGURE_FILE,
    dpi=400,
    facecolor="white",
    edgecolor="white",
    transparent=False,
)

try:
    from PIL import Image

    with Image.open(FIGURE_FILE) as image_file:
        if image_file.mode != "RGB":
            image_file.convert("RGB").save(FIGURE_FILE, dpi=(400, 400))
except ImportError as exc:
    raise ImportError(
        "Pillow is required to ensure RGB PNG output. Install it with "
        "'pip install pillow' or 'conda install pillow'."
    ) from exc

plt.show()

print(f"Plot data: {data_source}")
print(f"Saved figure: {FIGURE_FILE}")
print(f"Plotting cache: {CACHE_FILE}")
