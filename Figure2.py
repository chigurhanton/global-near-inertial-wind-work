# -*- coding: utf-8 -*-
"""Generate Figure 2 from source data or a reusable plotting-data cache.

Designed for execution as a single cell in a VS Code Jupyter notebook.
The notebook working directory is treated as the project root.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy import stats


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

DATA_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
CACHE_FILE = CACHE_DIR / "Fig2_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "Figure2.png"

POWER_VAR = "power_input"
YEAR_START = 1979
YEAR_END = 2025
CACHE_VERSION = 1
RECOMPUTE_CACHE = False

R_EARTH = 6_371_000.0
TREND_UNIT_TEXT = r"(mW m$^{-2}$ decade$^{-1}$)"

# North Pacific high-mean contour settings.
HIGH_VALUE_THRESHOLD = 1.8  # mW m^-2
CONTOUR_COLOR = "#FC00FC"
CONTOUR_STYLE = "--"
CONTOUR_WIDTH = 1.6

LAT_BIN_EDGES = np.arange(-90.0, 90.5, 0.5)


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
def compute_grid_area(ds, lat_name="latitude", lon_name="longitude"):
    """Compute regular-grid spherical cell area as a latitude-dependent array."""
    lat = ds[lat_name]
    lon = ds[lon_name]

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Latitude and longitude coordinates must each contain at least two points.")

    d_lat = float(np.abs(lat.values[1] - lat.values[0]))
    d_lon = float(np.abs(lon.values[1] - lon.values[0]))

    return (
        R_EARTH ** 2
        * np.deg2rad(d_lat)
        * np.deg2rad(d_lon)
        * np.cos(np.deg2rad(lat))
    )


def get_robinson_y(latitudes):
    """Return the Robinson-projection y coordinates for a list of latitudes."""
    latitudes = np.asarray(latitudes, dtype=float)
    proj = ccrs.Robinson(central_longitude=180)
    x_coords = np.full_like(latitudes, 180.0)
    pts = proj.transform_points(ccrs.PlateCarree(), x_coords, latitudes)
    return pts[:, 1]


def calc_pvalue(x, y):
    """Calculate the linear-trend P value at one grid cell."""
    if np.all(np.isnan(y)):
        return np.nan

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        return np.nan

    _, _, _, p_value, _ = stats.linregress(x[mask], y[mask])
    return p_value



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
    """Check whether an existing cache matches the current Figure 2 settings."""
    required_keys = {
        "cache_version",
        "year_start",
        "year_end",
        "high_value_threshold",
        "lat_bin_edges",
        "lon_cyclic",
        "lat",
        "trend_cyclic",
        "trend_limit",
        "pval_cyclic",
        "mean_cyclic_npac",
        "c_pos_arr",
        "c_neg_arr",
        "c_net_arr_plot",
    }

    if not required_keys.issubset(cache.files):
        return False

    return (
        int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["year_start"]) == YEAR_START
        and int(cache["year_end"]) == YEAR_END
        and np.isclose(float(cache["high_value_threshold"]), HIGH_VALUE_THRESHOLD)
        and np.array_equal(cache["lat_bin_edges"], LAT_BIN_EDGES)
    )


def compute_plot_data(source_file):
    """Compute all data required to reproduce Figure 2."""
    with xr.open_dataset(source_file) as ds:
        try:
            ds = ds.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
        except Exception:
            pass

        if POWER_VAR not in ds:
            raise KeyError(f"Variable '{POWER_VAR}' was not found in {source_file.name}.")

        lat = ds.latitude.values.astype(float)
        lon = ds.longitude.values.astype(float)
        area_weights = compute_grid_area(ds)

        # Mean state used for the North Pacific contour.
        mean_map = ds[POWER_VAR].mean(dim="time", skipna=True) * 1000.0
        mean_cyclic, lon_cyclic = add_cyclic_point(mean_map.values, coord=lon)

        # Spatial linear trend.
        n_months = int(ds.sizes["time"])
        ds = ds.assign_coords(time_years=("time", np.arange(n_months, dtype=float) / 12.0))
        poly = ds[POWER_VAR].polyfit(dim="time_years", deg=1, skipna=True)
        trend_map = poly.polyfit_coefficients.sel(degree=1) * 1e3 * 10.0
        trend_cyclic, _ = add_cyclic_point(trend_map.values, coord=lon)
        trend_limit = float(np.nanpercentile(np.abs(trend_map.values), 99))

        # Pixel-wise significance.
        pval_map = xr.apply_ufunc(
            calc_pvalue,
            ds.time_years,
            ds[POWER_VAR],
            input_core_dims=[["time"], ["time"]],
            vectorize=True,
            output_dtypes=[float],
        )
        pval_cyclic, _ = add_cyclic_point(pval_map.values, coord=lon)

        # Latitudinal integrated contributions (GW decade^-1).
        trend_w_m2_dec = trend_map / 1000.0
        trend_gw_dec = (trend_w_m2_dec * area_weights) / 1e9
        lat_grid = trend_gw_dec.latitude

        c_pos_list = []
        c_neg_list = []
        c_net_list = []
        for i in range(len(LAT_BIN_EDGES) - 1):
            lat_mask = (lat_grid >= LAT_BIN_EDGES[i]) & (lat_grid < LAT_BIN_EDGES[i + 1])
            band = trend_gw_dec.where(lat_mask)
            c_pos = band.where(band > 0).sum(skipna=True).item()
            c_neg = band.where(band < 0).sum(skipna=True).item()
            c_pos_list.append(c_pos)
            c_neg_list.append(c_neg)
            c_net_list.append(c_pos + c_neg)

        c_pos_arr = np.asarray(c_pos_list, dtype=float)
        c_neg_arr = np.asarray(c_neg_list, dtype=float)
        c_net_arr = np.asarray(c_net_list, dtype=float)

        c_net_arr_plot = np.copy(c_net_arr)
        no_data_mask = (np.abs(c_pos_arr) < 1e-6) & (np.abs(c_neg_arr) < 1e-6)
        c_net_arr_plot[no_data_mask] = np.nan

        # Mean-state contour only inside the North Pacific box.
        lon_mesh, lat_mesh = np.meshgrid(lon_cyclic, lat)
        lon_mod = (lon_mesh + 180.0) % 360.0 - 180.0
        mask_npac = (
            (lat_mesh >= 30.0)
            & (lat_mesh <= 60.0)
            & ((lon_mod >= 120.0) | (lon_mod <= -120.0))
        )
        mean_cyclic_npac = np.where(mask_npac, mean_cyclic, np.nan)

    return {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "year_start": np.array(YEAR_START, dtype=np.int16),
        "year_end": np.array(YEAR_END, dtype=np.int16),
        "high_value_threshold": np.array(HIGH_VALUE_THRESHOLD, dtype=float),
        "lat_bin_edges": LAT_BIN_EDGES,
        "lon_cyclic": lon_cyclic,
        "lat": lat,
        "trend_cyclic": trend_cyclic,
        "trend_limit": np.array(trend_limit, dtype=float),
        "pval_cyclic": pval_cyclic,
        "mean_cyclic_npac": mean_cyclic_npac,
        "c_pos_arr": c_pos_arr,
        "c_neg_arr": c_neg_arr,
        "c_net_arr_plot": c_net_arr_plot,
    }


def load_or_build_plot_data():
    """Load a compatible cache or rebuild it from the source NetCDF file."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            if cache_is_compatible(cache):
                return {key: np.array(cache[key]) for key in cache.files}, "cache"

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Figure 2 cannot be generated because neither a compatible plotting "
            "cache nor the source NetCDF file is available.\n\n"
            f"Expected plotting cache:\n  {CACHE_FILE}\n\n"
            f"Expected source data for the first run:\n  {DATA_FILE}\n\n"
            "Place the source NetCDF file in the project 'data' folder for the "
            "first run, or place a compatible Fig2_plot_data.npz file in 'cache'."
        )

    plot_data = compute_plot_data(DATA_FILE)
    np.savez_compressed(CACHE_FILE, **plot_data)
    return plot_data, "source data"


# =============================================================================
# 4. Load or compute plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()

lon_cyclic = data["lon_cyclic"].astype(float)
lat = data["lat"].astype(float)
trend_cyclic = data["trend_cyclic"].astype(float)
trend_limit = float(data["trend_limit"])
pval_cyclic = data["pval_cyclic"].astype(float)
mean_cyclic_npac = data["mean_cyclic_npac"].astype(float)
c_pos_arr = data["c_pos_arr"].astype(float)
c_neg_arr = data["c_neg_arr"].astype(float)
c_net_arr_plot = data["c_net_arr_plot"].astype(float)

lat_bin_edges = data["lat_bin_edges"].astype(float)
y_centers = 0.5 * (lat_bin_edges[:-1] + lat_bin_edges[1:])
y_centers_rob = get_robinson_y(y_centers)

# Remove only the endpoint labels 80°S and 80°N, keep the others.
target_lats = np.array([-60, -40, -20, 0, 20, 40, 60], dtype=float)
target_y = get_robinson_y(target_lats)
lat_labels_ns = [
    f"{int(abs(lat_value))}°S" if lat_value < 0 else (f"{int(lat_value)}°N" if lat_value > 0 else "0°")
    for lat_value in target_lats
]


# =============================================================================
# 5. Figure 2
# =============================================================================
# Width ≈ 183 mm, consistent with Figure 1.
fig = plt.figure(figsize=(7.2, 4.35), facecolor="white")

pos_map = [0.06, 0.18, 0.64, 0.70]
pos_zonal = [0.75, 0.18, 0.18, 0.70]
pos_cbar = [0.21, 0.11, 0.34, 0.016]

# Panel a: spatial trend map.
ax_map = fig.add_axes(pos_map, projection=ccrs.Robinson(central_longitude=180))

map_image = ax_map.pcolormesh(
    lon_cyclic,
    lat,
    trend_cyclic,
    transform=ccrs.PlateCarree(),
    cmap=plt.cm.RdBu_r,
    vmin=-trend_limit,
    vmax=trend_limit,
    shading="auto",
    rasterized=True,
    zorder=1,
)

# Significance hatching for P < 0.05.
pval_filled = np.nan_to_num(pval_cyclic, nan=1.0)
ax_map.contourf(
    lon_cyclic,
    lat,
    pval_filled,
    levels=[0.0, 0.05],
    transform=ccrs.PlateCarree(),
    colors="none",
    hatches=["///"],
    alpha=0.0,
    zorder=2,
)

# North Pacific high-mean contour.
ax_map.contour(
    lon_cyclic,
    lat,
    mean_cyclic_npac,
    levels=[HIGH_VALUE_THRESHOLD],
    colors=CONTOUR_COLOR,
    linewidths=CONTOUR_WIDTH,
    linestyles=CONTOUR_STYLE,
    transform=ccrs.PlateCarree(),
    zorder=3,
)

ax_map.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=4)
ax_map.coastlines(linewidth=0.5, zorder=5)
ax_map.set_global()
ax_map.set_title("a", loc="left", fontweight="bold", pad=4)

# Draw gridlines without automatic labels. Latitude labels are added manually
# on the left only, which avoids duplicate labels on Robinson-map boundaries.
gl = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.7,
    color="#666666",
    alpha=0.45,
    linestyle="--",
)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

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

cbar_ax = fig.add_axes(pos_cbar)
colorbar = fig.colorbar(map_image, cax=cbar_ax, orientation="horizontal")
colorbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
colorbar.outline.set_linewidth(0.8)
colorbar.set_label(TREND_UNIT_TEXT, fontsize=8.0, labelpad=3)

# Panel b: latitudinal integrated contributions.
ax_zonal = fig.add_axes(pos_zonal)

ax_zonal.fill_betweenx(
    y_centers_rob,
    0,
    c_pos_arr,
    color="#D62728",
    alpha=0.45,
    label="Positive",
)
ax_zonal.fill_betweenx(
    y_centers_rob,
    0,
    c_neg_arr,
    color="#1F77B4",
    alpha=0.45,
    label="Negative",
)
ax_zonal.plot(
    c_net_arr_plot,
    y_centers_rob,
    color="black",
    linewidth=1.2,
    zorder=3,
    label="Net",
)

max_c = float(np.nanmax(np.abs(np.concatenate([c_pos_arr, c_neg_arr]))))
if not np.isfinite(max_c) or max_c == 0.0:
    max_c = 1.0
ax_zonal.set_xlim(-1.1 * max_c, 1.1 * max_c)
ax_zonal.axvline(0.0, color="#808080", linestyle="--", linewidth=0.9, zorder=1)

ax_zonal.set_yticks(target_y)
ax_zonal.set_yticklabels(lat_labels_ns)
ax_zonal.set_ylim(get_robinson_y(np.array([-90.0, 90.0], dtype=float)))
ax_zonal.yaxis.tick_right()
ax_zonal.yaxis.set_label_position("right")
ax_zonal.set_ylabel("Latitude")

ax_zonal.set_xlabel("Integrated contribution\n" + r"(GW decade$^{-1}$)")
ax_zonal.tick_params(axis="x", labelsize=7.2, width=0.8, length=3)
ax_zonal.tick_params(axis="y", labelsize=7.2, width=0.8, length=3)
ax_zonal.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax_zonal.set_title("b", loc="left", fontweight="bold", pad=4)
ax_zonal.grid(True, linestyle=":", linewidth=0.7, color="#B0B0B0", alpha=0.45)
ax_zonal.legend(
    loc="upper left",
    frameon=False,
    handlelength=1.5,
    borderaxespad=0.25,
    labelspacing=0.3,
)

# Align panel-b height exactly with the map panel after projection rendering.
fig.canvas.draw()
map_pos = ax_map.get_position()
zonal_pos = ax_zonal.get_position()
ax_zonal.set_position([zonal_pos.x0, map_pos.y0, zonal_pos.width, map_pos.height])

cbar_pos = cbar_ax.get_position()
cbar_ax.set_position([cbar_pos.x0, map_pos.y0 - 0.07, cbar_pos.width, cbar_pos.height])


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
