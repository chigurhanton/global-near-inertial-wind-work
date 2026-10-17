# -*- coding: utf-8 -*-
"""Generate the revised Figure 3 using the NEAR-INERTIAL WIND-STRESS product.

Panels
------
a  Trend in near-inertial-band wind-stress variance
      source: resonant_ni_wind_stress_variance, from
              calculate_wind_stress_ni_total_variance_1979_2025.py
      (the previous version of panel a used near-inertial wind-SPEED variance,
       resonant_ni_wind_variance)
b  Trend in mixed-layer depth (MLD)
c  Relative wind / MLD importance map:
      D = |beta_climH| / (|beta_climH| + |beta_MLD|)
   where
      beta_climH = NIWW trend from varying wind + climatological monthly MLD
      beta_MLD   = beta_full - beta_climH
      beta_full  = NIWW trend from varying wind + evolving MLD

Diagonal hatching in panels a and b denotes P < 0.05. Cross-hatching in panel c
marks opposite signs of beta_full and beta_climH, without a significance filter.
The marker meaning is described in the caption; no marker legend is drawn.
Panel a retains the 10^-4 display units and percentile scaling of this version.
Panel c retains its D colour field, colour scale and blank colourbar-end labels.
Both trend fields already exist in the compatible plotting cache. Exactly zero
trends are classified separately and are not marked as sign reversals.

NOTE ON THE OCEAN MASK
     The wind-stress product is defined over the full 60S-60N grid INCLUDING
     LAND, whereas the older wind-speed product and the wind-work products are
     ocean-only. Panel a therefore applies an ocean mask taken from the finite
     values of power_input in niw_power_input_1979_2025_Full.nc (verified
     time-invariant, same 0.25 degree grid). Without it, land would
     contaminate the area-weighted statistics printed at the end.

The total wind-stress variance trend is NOT part of this figure; it is drawn
separately by FigureS_windstress_total.py for the appendix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy.stats import linregress

# Optional readers for MLD .mat
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

# Input files
NC_STRESS_PATH = DATA_DIR / "global_wind_stress_ni_total_variance_1979_2025.nc"
OCEAN_MASK_PATH = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
VARYING_MLD_FILE = DATA_DIR / "niw_power_input_1979_2025_Full.nc"
CLIMATOLOGICAL_MLD_FILE = DATA_DIR / "deeplearning_climmld_niw_power_input_1979_2025.nc"
MLD_FILE = DATA_DIR / "mld_all_1979to2025.mat"

# Output
CACHE_FILE = CACHE_DIR / "Fig3_windstress_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "Figure3.png"

# Variables
NI_VARIABLE = "resonant_ni_wind_stress_variance"
TOTAL_VARIABLE = "total_wind_stress_variance"
MASK_VARIABLE = "power_input"
WINDWORK_VAR = "power_input"
MLD_VAR = "mld_all"

# Time range
START_YEAR = 1979
END_YEAR = 2025
YEAR_SLICE = slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")

# Analysis mask
MIN_ABS_LAT = 3.0
MAX_ABS_LAT = 60.0

# Cache
CACHE_VERSION = 1
RECOMPUTE_CACHE = False

# Figure layout: same compact three-panel format as Figure3_new.py
FIG_WIDTH = 5.4
FIG_HEIGHT = 7.8

# Significance (panel-a/b hatching only)
P_THRESHOLD = 0.05

# Color scaling
MAP_A_PERCENTILE = 98.0
MAP_B_PERCENTILE = 98.0

# Relative-importance denominator threshold
DENOM_EPS = 1e-12

# Panel-a values are displayed in units of 10^-4; cached values stay in SI units.
TREND_PLOT_SCALE = 1.0e4
TREND_UNIT_TEXT = r"(10$^{-4}$ N$^{2}$ m$^{-4}$ decade$^{-1}$)"


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
# 3. Utilities
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


def standardize_spatiotemporal(da):
    """Standardize coords and crop to analysis years."""
    da = rename_latlon(da)
    required = {"time", "lat", "lon"}
    available = set(da.coords) | set(da.dims)
    missing = required.difference(available)
    if missing:
        raise KeyError(
            f"Missing coordinates/dimensions: {sorted(missing)}; found: {list(available)}"
        )
    da = da.sel(time=YEAR_SLICE)
    da = da.sortby("lat").sortby("lon").sortby("time")
    return da


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
    """Keep 3-60 degrees latitude in both hemispheres."""
    lat_abs = np.abs(da["lat"])
    return da.where((lat_abs >= MIN_ABS_LAT) & (lat_abs <= MAX_ABS_LAT))


def ensure_yearly_time(da):
    """Keep annual data as-is; otherwise average within calendar year."""
    da = da.sortby("time")
    years = pd.DatetimeIndex(pd.to_datetime(da.time.values)).year
    if len(years) == len(np.unique(years)):
        return da
    return da.resample(time="1YE").mean(dim="time", skipna=True)


def normalize_annual_time(da, name):
    """Reset annual timestamps to Jan 1 of each year."""
    years = pd.DatetimeIndex(pd.to_datetime(da.time.values)).year
    unique, counts = np.unique(years, return_counts=True)
    duplicated = unique[counts > 1]
    if duplicated.size:
        raise ValueError(f"{name} still has duplicated years: {duplicated.tolist()}")
    return da.assign_coords(time=pd.to_datetime([f"{y}-01-01" for y in years])).sortby("time")


def annual_mean_field(da):
    """Simple arithmetic mean within each calendar year."""
    return da.resample(time="1YE").mean(dim="time", skipna=True)


def calc_trend_and_pvalue(annual_da):
    """Grid-wise linear trend and p value; slope is per year."""
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


def read_ocean_mask(reference_grid):
    """Time-invariant ocean mask from the finite values of power_input.

    power_input exists only where BOTH wind work and MLD are defined, i.e. over
    the ocean. The mask is verified to be identical across the record, so a
    single slice is representative.
    """
    with xr.open_dataset(OCEAN_MASK_PATH) as ds_mask:
        if MASK_VARIABLE not in ds_mask.data_vars:
            raise KeyError(
                f"Variable '{MASK_VARIABLE}' not found in {OCEAN_MASK_PATH.name}. "
                f"Available variables: {list(ds_mask.data_vars)}"
            )
        raw = rename_latlon(ds_mask[MASK_VARIABLE])
        checks = [
            raw.isel(time=0).notnull().values,
            raw.isel(time=len(raw["time"]) // 2).notnull().values,
            raw.isel(time=-1).notnull().values,
        ]
        if not (np.array_equal(checks[0], checks[1]) and np.array_equal(checks[0], checks[2])):
            raise ValueError(
                "The ocean mask derived from power_input is not time-invariant; "
                "inspect the source file before using it as a mask."
            )
        mask = xr.DataArray(
            checks[0].astype(np.float32),
            coords={"lat": raw["lat"], "lon": raw["lon"]},
            dims=("lat", "lon"),
        )

    mask = harmonize_longitude_to_reference(mask, reference_grid)
    mask = mask.interp_like(reference_grid, method="nearest") > 0.5
    return mask


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


def robust_symmetric_limit(da, percentile):
    """Symmetric color limit from a percentile of the absolute values."""
    values = np.asarray(da.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def calculate_dominance_area_fraction(
    relative_importance,
    pval_full=None,
    slope_full=None,
    significant_only=False,
    positive_only=False,
    threshold=0.5,
):
    """Area-weighted fractions of wind- and MLD-dominated regions (percent)."""
    D = relative_importance
    valid = np.isfinite(D)

    if significant_only:
        if pval_full is None:
            raise ValueError("pval_full is required when significant_only=True")
        valid = valid & np.isfinite(pval_full) & (pval_full < 0.05)

    if positive_only:
        if slope_full is None:
            raise ValueError("slope_full is required when positive_only=True")
        valid = valid & np.isfinite(slope_full) & (slope_full > 0)

    area_weight = xr.DataArray(
        np.cos(np.deg2rad(D["lat"])),
        coords={"lat": D["lat"]},
        dims=("lat",),
    ).broadcast_like(D)

    wind_mask = valid & (D > threshold)
    mld_mask = valid & (D < threshold)
    balanced_mask = valid & np.isclose(D, threshold, atol=1e-10)

    total_area = area_weight.where(valid).sum(skipna=True).item()
    if total_area <= 0:
        return {"wind": np.nan, "mld": np.nan, "balanced": np.nan}

    wind_area = area_weight.where(wind_mask).sum(skipna=True).item()
    mld_area = area_weight.where(mld_mask).sum(skipna=True).item()
    balanced_area = area_weight.where(balanced_mask).sum(skipna=True).item()

    return {
        "wind": 100.0 * wind_area / total_area,
        "mld": 100.0 * mld_area / total_area,
        "balanced": 100.0 * balanced_area / total_area,
    }


def classify_trend_signs(slope_full, slope_climh, relative_importance):
    """Classify the two NIWW trend signs on panel c's valid ocean domain.

    Opposite strictly nonzero signs define a reversal. No p-value filter and
    no near-zero tolerance are applied. A zero in either experiment is a
    separate category, not a reversal or a retained nonzero sign.
    """
    full, climh, domain = xr.align(
        slope_full, slope_climh, relative_importance, join="exact"
    )
    valid = np.isfinite(full) & np.isfinite(climh) & np.isfinite(domain)
    reversed_sign = valid & (
        ((full > 0) & (climh < 0)) | ((full < 0) & (climh > 0))
    )
    same_sign = valid & (
        ((full > 0) & (climh > 0)) | ((full < 0) & (climh < 0))
    )
    zero_trend = valid & ((full == 0) | (climh == 0))
    return {
        "valid": valid,
        "reversed": reversed_sign,
        "same": same_sign,
        "zero": zero_trend,
    }

def calculate_trend_sign_area_fractions(sign_masks):
    """Cos(latitude)-weighted percentages over the same valid domain as panel c.

    Longitude is uniform in the input grid, so the common longitude width
    cancels. Fractions use original grid cells, never the plotting-only cyclic
    longitude duplicate. Reversed + same + zero sum to 100 percent.
    """
    valid = sign_masks["valid"]
    area_weight = xr.DataArray(
        np.cos(np.deg2rad(valid["lat"])),
        coords={"lat": valid["lat"]},
        dims=("lat",),
    ).broadcast_like(valid)
    total_area = float(area_weight.where(valid).sum(skipna=True).item())
    if total_area <= 0:
        return {key: np.nan for key in ("reversed", "same", "zero", "not_reversed")}
    fractions = {
        key: 100.0 * float(area_weight.where(sign_masks[key]).sum(skipna=True).item())
        / total_area
        for key in ("reversed", "same", "zero")
    }
    fractions["not_reversed"] = fractions["same"] + fractions["zero"]
    return fractions


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
    pvalue_da,
    panel_label,
    cmap,
    unit_text=None,
    percentile=98.0,
    show_bottom=False,
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    hatch_mask=None,
    hatch_pattern="///",
):
    """Plot one map with significance or an explicit hatch mask; no hatch legend."""
    data = np.asarray(da.values, dtype=float)
    if hatch_mask is None:
        pval = np.asarray(pvalue_da.values, dtype=float)
        hatch_values = np.isfinite(pval) & (pval < P_THRESHOLD)
    else:
        _, hatch_mask = xr.align(da, hatch_mask, join="exact")
        hatch_values = np.asarray(hatch_mask.values, dtype=bool)
    hatch_values = hatch_values & np.isfinite(data)

    cyclic_data, cyclic_lon = add_cyclic_point(data, coord=da["lon"].values)
    cyclic_hatch = add_cyclic_point(hatch_values.astype(float), coord=da["lon"].values)[0]

    if vmin is None or vmax is None:
        vmax_plot = robust_symmetric_limit(da, percentile)
        vmin_plot = -vmax_plot
    else:
        vmin_plot = vmin
        vmax_plot = vmax

    image = ax.pcolormesh(
        cyclic_lon,
        da["lat"].values,
        cyclic_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        shading="auto",
        zorder=1,
        rasterized=True,
    )

    if np.any(hatch_values):
        ax.contourf(
            cyclic_lon,
            da["lat"].values,
            cyclic_hatch,
            levels=[0.5, 1.5],
            colors="none",
            hatches=[hatch_pattern],
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", edgecolor="none", zorder=4)
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=8)
    ax.set_global()
    add_map_gridlines(ax, show_bottom)
    ax.set_title(panel_label, loc="left", fontweight="bold", pad=4)

    colorbar = fig3.colorbar(
        image,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=cbar_ticks,
    )
    colorbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
    colorbar.outline.set_linewidth(0.8)

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

    if panel_label == "c":
        colorbar.ax.text(
            -0.04,
            0.50,
            " ",
            transform=colorbar.ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.0,
            clip_on=False,
        )
        colorbar.ax.text(
            1.04,
            0.50,
            " ",
            transform=colorbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
            clip_on=False,
        )


# =============================================================================
# 4. Cache generation
# =============================================================================
def cache_is_compatible(cache):
    required = {
        "cache_version",
        "start_year",
        "end_year",
        "min_abs_lat",
        "max_abs_lat",
        "lat",
        "lon",
        "slope_ni_dec",
        "pval_ni",
        "slope_total_dec",
        "pval_total",
        "slope_mld_dec",
        "pval_mld",
        "slope_full_dec",
        "pval_full",
        "slope_climh_dec",
        "slope_mld_mod_dec",
        "relative_importance",
    }
    return (
        required.issubset(cache.files)
        and int(cache["cache_version"]) == CACHE_VERSION
        and int(cache["start_year"]) == START_YEAR
        and int(cache["end_year"]) == END_YEAR
        and np.isclose(float(cache["min_abs_lat"]), MIN_ABS_LAT)
        and np.isclose(float(cache["max_abs_lat"]), MAX_ABS_LAT)
    )


def compute_plot_data():
    """Compute all products needed for the revised Figure 3."""
    required_files = [
        NC_STRESS_PATH,
        OCEAN_MASK_PATH,
        CLIMATOLOGICAL_MLD_FILE,
        MLD_FILE,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 3 cannot be generated because required source files are missing:\n\n"
            + "\n".join(str(path) for path in missing)
        )

    # -------------------------------------------------------------------------
    # A. Near-inertial WIND-STRESS variance trend (ocean only)
    # -------------------------------------------------------------------------
    with xr.open_dataset(NC_STRESS_PATH) as ds_stress:
        for variable_name in (NI_VARIABLE, TOTAL_VARIABLE):
            if variable_name not in ds_stress.data_vars:
                raise KeyError(
                    f"Variable '{variable_name}' not found in {NC_STRESS_PATH.name}. "
                    f"Available variables: {list(ds_stress.data_vars)}"
                )
        ni_raw = standardize_spatiotemporal(ds_stress[NI_VARIABLE])
        total_raw = standardize_spatiotemporal(ds_stress[TOTAL_VARIABLE])

    reference_grid = ni_raw.isel(time=0, drop=True)
    ocean_mask = read_ocean_mask(reference_grid)

    ni = apply_analysis_mask(ni_raw).where(ocean_mask)
    total = apply_analysis_mask(total_raw).where(ocean_mask)

    ni_ann = normalize_annual_time(ensure_yearly_time(ni), "NI wind-stress variance")
    total_ann = normalize_annual_time(ensure_yearly_time(total), "total wind-stress variance")

    slope_ni_year, pval_ni = calc_trend_and_pvalue(ni_ann)
    slope_total_year, pval_total = calc_trend_and_pvalue(total_ann)

    # Stored in SI units (per decade); TREND_PLOT_SCALE is applied only at draw time.
    slope_ni_dec = apply_analysis_mask(slope_ni_year * 10.0)
    slope_total_dec = apply_analysis_mask(slope_total_year * 10.0)
    pval_ni = apply_analysis_mask(pval_ni)
    pval_total = apply_analysis_mask(pval_total)

    # -------------------------------------------------------------------------
    # B. MLD trend
    # -------------------------------------------------------------------------
    mld = read_mld_as_time_lat_lon(MLD_FILE)
    mld = harmonize_longitude_to_reference(mld, reference_grid)
    mld = mld.interp_like(reference_grid, method="nearest")
    mld_ann = normalize_annual_time(annual_mean_field(mld), "MLD")
    slope_mld_year, pval_mld = calc_trend_and_pvalue(mld_ann)
    slope_mld_dec = apply_analysis_mask(slope_mld_year * 10.0)
    pval_mld = apply_analysis_mask(pval_mld)

    # -------------------------------------------------------------------------
    # C. Full vs climatological-MLD NIWW trends -> relative importance
    # -------------------------------------------------------------------------
    with xr.open_dataset(VARYING_MLD_FILE) as ds_full, \
            xr.open_dataset(CLIMATOLOGICAL_MLD_FILE) as ds_climh:
        if WINDWORK_VAR not in ds_full.data_vars:
            raise KeyError(
                f"Variable '{WINDWORK_VAR}' not found in {VARYING_MLD_FILE.name}. "
                f"Available variables: {list(ds_full.data_vars)}"
            )
        if WINDWORK_VAR not in ds_climh.data_vars:
            raise KeyError(
                f"Variable '{WINDWORK_VAR}' not found in {CLIMATOLOGICAL_MLD_FILE.name}. "
                f"Available variables: {list(ds_climh.data_vars)}"
            )
        full_ww = standardize_spatiotemporal(ds_full[WINDWORK_VAR])
        climh_ww = standardize_spatiotemporal(ds_climh[WINDWORK_VAR])

    full_ww = harmonize_longitude_to_reference(full_ww, reference_grid)
    climh_ww = harmonize_longitude_to_reference(climh_ww, reference_grid)

    full_ww = full_ww.interp_like(reference_grid, method="nearest")
    climh_ww = climh_ww.interp_like(reference_grid, method="nearest")

    full_ww = apply_analysis_mask(full_ww)
    climh_ww = apply_analysis_mask(climh_ww)

    full_ann = normalize_annual_time(annual_mean_field(full_ww), "full NIWW")
    climh_ann = normalize_annual_time(annual_mean_field(climh_ww), "climatological-MLD NIWW")

    slope_full_year, pval_full = calc_trend_and_pvalue(full_ann)
    slope_climh_year, _ = calc_trend_and_pvalue(climh_ann)

    slope_full_dec = apply_analysis_mask(slope_full_year * 10.0)
    slope_climh_dec = apply_analysis_mask(slope_climh_year * 10.0)
    pval_full = apply_analysis_mask(pval_full)

    # MLD modulation term
    slope_mld_mod_dec = slope_full_dec - slope_climh_dec

    # D = |beta_climH| / (|beta_climH| + |beta_MLD|)
    denom = np.abs(slope_climh_dec) + np.abs(slope_mld_mod_dec)
    relative_importance = xr.where(
        denom > DENOM_EPS,
        np.abs(slope_climh_dec) / denom,
        np.nan,
    )
    relative_importance = apply_analysis_mask(relative_importance)

    plot_data = {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int16),
        "start_year": np.array(START_YEAR, dtype=np.int16),
        "end_year": np.array(END_YEAR, dtype=np.int16),
        "min_abs_lat": np.array(MIN_ABS_LAT, dtype=float),
        "max_abs_lat": np.array(MAX_ABS_LAT, dtype=float),
        "lat": slope_ni_dec["lat"].values,
        "lon": slope_ni_dec["lon"].values,
        "slope_ni_dec": slope_ni_dec.values,
        "pval_ni": pval_ni.values,
        "slope_total_dec": slope_total_dec.values,
        "pval_total": pval_total.values,
        "slope_mld_dec": slope_mld_dec.values,
        "pval_mld": pval_mld.values,
        "slope_full_dec": slope_full_dec.values,
        "pval_full": pval_full.values,
        "slope_climh_dec": slope_climh_dec.values,
        "slope_mld_mod_dec": slope_mld_mod_dec.values,
        "relative_importance": relative_importance.values,
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
# 5. Load plotting data
# =============================================================================
data, data_source = load_or_build_plot_data()
lat = data["lat"]
lon = data["lon"]

slope_ni_dec = reconstruct_da(data["slope_ni_dec"], lat, lon, "slope_ni_dec")
pval_ni = reconstruct_da(data["pval_ni"], lat, lon, "pval_ni")

slope_mld_dec = reconstruct_da(data["slope_mld_dec"], lat, lon, "slope_mld_dec")
pval_mld = reconstruct_da(data["pval_mld"], lat, lon, "pval_mld")

slope_full_dec = reconstruct_da(data["slope_full_dec"], lat, lon, "slope_full_dec")
pval_full = reconstruct_da(data["pval_full"], lat, lon, "pval_full")

slope_climh_dec = reconstruct_da(data["slope_climh_dec"], lat, lon, "slope_climh_dec")
slope_mld_mod_dec = reconstruct_da(data["slope_mld_mod_dec"], lat, lon, "slope_mld_mod_dec")

relative_importance = reconstruct_da(data["relative_importance"], lat, lon, "relative_importance")

# Opposite experiment trend signs, independent of P values and of D < 0.5.
trend_sign_masks = classify_trend_signs(
    slope_full_dec, slope_climh_dec, relative_importance
)
sign_reversal_mask = trend_sign_masks["reversed"]

# Display scaling only: the cached panel-a field stays in SI units.
slope_ni_plot = slope_ni_dec * TREND_PLOT_SCALE
# Panel b keeps the colour scale of Figure3_new.py (metres per decade).
slope_mld_plot = slope_mld_dec


# =============================================================================
# 6. Colormaps
# =============================================================================
CMAP_A = "PuOr_r"
CMAP_B = "PiYG_r"
CMAP_C = LinearSegmentedColormap.from_list(
    "mld_to_wind",
    [
        (0.00, "#D44623"),
        (0.50, "#F2F2F2"),
        (1.00, "#3182bd"),
    ],
    N=256,
)


# =============================================================================
# 7. Figure 3
# =============================================================================
fig3 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
projection = ccrs.Robinson(central_longitude=180)

map_positions = [
    [0.08, 0.705, 0.84, 0.235],
    [0.08, 0.395, 0.84, 0.235],
    [0.08, 0.090, 0.84, 0.235],
]

cbar_positions = [
    [0.29, 0.675, 0.42, 0.012],
    [0.29, 0.365, 0.42, 0.012],
    [0.29, 0.045, 0.42, 0.012],
]

axes = [fig3.add_axes(position, projection=projection) for position in map_positions]
cbar_axes = [fig3.add_axes(position) for position in cbar_positions]

# Panel a: near-inertial wind-stress variance trend
plot_panel(
    ax=axes[0],
    cbar_ax=cbar_axes[0],
    da=slope_ni_plot,
    pvalue_da=pval_ni,
    panel_label="a",
    cmap=CMAP_A,
    unit_text=TREND_UNIT_TEXT,
    percentile=MAP_A_PERCENTILE,
    show_bottom=False,
)

# Panel b: MLD trend
plot_panel(
    ax=axes[1],
    cbar_ax=cbar_axes[1],
    da=slope_mld_plot,
    pvalue_da=pval_mld,
    panel_label="b",
    cmap=CMAP_B,
    unit_text=r"(m decade$^{-1}$)",
    percentile=MAP_B_PERCENTILE,
    show_bottom=False,
)

# Panel c: relative wind / MLD importance
plot_panel(
    ax=axes[2],
    cbar_ax=cbar_axes[2],
    da=relative_importance,
    pvalue_da=None,  # Cross-hatching does not use a significance threshold.
    panel_label="c",
    cmap=CMAP_C,
    unit_text=None,
    show_bottom=True,
    vmin=0.0,
    vmax=1.0,
    cbar_ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
    hatch_mask=sign_reversal_mask,
    hatch_pattern="xx",
)

fig3.savefig(
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

print(f"Plot data: {data_source}")
print(f"Saved figure: {FIGURE_FILE}")
print(f"Plotting cache: {CACHE_FILE}")


# =============================================================================
# 8. Diagnostics
# =============================================================================
print("\n" + "=" * 68)
print("PANEL A: NEAR-INERTIAL WIND-STRESS VARIANCE TREND (ocean, 3-60 deg)")
print("=" * 68)

valid_cells = int(np.isfinite(slope_ni_dec).sum())
print(f"\nValid ocean grid cells            : {valid_cells}")
print(f"Significant cells (P < 0.05)      : "
      f"{100.0 * float((np.isfinite(pval_ni) & (pval_ni < P_THRESHOLD)).sum()) / valid_cells:.1f}%")
print(f"Positive-trend cells              : "
      f"{100.0 * float((slope_ni_dec > 0).sum()) / valid_cells:.1f}%")

years = np.arange(START_YEAR, END_YEAR + 1, dtype=float)
with xr.open_dataset(NC_STRESS_PATH) as ds_stress:
    ni_field = standardize_spatiotemporal(ds_stress[NI_VARIABLE])
    ocean_mask = read_ocean_mask(ni_field.isel(time=0, drop=True))
    ni_field = apply_analysis_mask(ni_field).where(ocean_mask)
    weights = np.cos(np.deg2rad(ni_field["lat"])).broadcast_like(ni_field).where(
        np.isfinite(ni_field))
    ni_series = (
        (ni_field * weights).sum(dim=["lat", "lon"]) / weights.sum(dim=["lat", "lon"])
    ).values.astype(float)

slope, intercept, _, p_value, _ = linregress(years, ni_series)
print(f"\nArea-weighted global mean of panel a field:")
print(f"   mean 1979-2025 : {np.nanmean(ni_series):.6g} N2 m-4")
print(f"   trend          : {slope:.6g} per year "
      f"({slope * 10.0:.6g} per decade), p = {p_value:.3g}")
print(f"   relative trend : {100.0 * slope / np.nanmean(ni_series):.3f} % per year")


# =============================================================================
# 9. Area-weighted wind / MLD dominance
# =============================================================================
frac_all = calculate_dominance_area_fraction(
    relative_importance,
    pval_full=pval_full,
    slope_full=slope_full_dec,
    significant_only=False,
    positive_only=False,
)
frac_sig = calculate_dominance_area_fraction(
    relative_importance,
    pval_full=pval_full,
    slope_full=slope_full_dec,
    significant_only=True,
    positive_only=False,
)
frac_sig_pos = calculate_dominance_area_fraction(
    relative_importance,
    pval_full=pval_full,
    slope_full=slope_full_dec,
    significant_only=True,
    positive_only=True,
)

print("\n" + "=" * 68)
print("AREA-WEIGHTED WIND / MLD DOMINANCE (panel c)")
print("=" * 68)
print("\nAll valid ocean regions:")
print(f"  Wind-dominated : {frac_all['wind']:.1f}%")
print(f"  MLD-dominated  : {frac_all['mld']:.1f}%")
print("\nRegions with significant total NIWW trend (P < 0.05):")
print(f"  Wind-dominated : {frac_sig['wind']:.1f}%")
print(f"  MLD-dominated  : {frac_sig['mld']:.1f}%")
print("\nRegions with significant positive NIWW trend (P < 0.05):")
print(f"  Wind-dominated : {frac_sig_pos['wind']:.1f}%")
print(f"  MLD-dominated  : {frac_sig_pos['mld']:.1f}%")
print("=" * 68)

# =============================================================================
# 10. Trend sign agreement: the panel-c hatching criterion (NO p-value filter)
# =============================================================================
sign_fractions = calculate_trend_sign_area_fractions(trend_sign_masks)
print("\n" + "=" * 68)
print("PANEL C: AREA-WEIGHTED TREND SIGN COMPARISON (NO P-VALUE FILTER)")
print("=" * 68)
print("Domain: valid ocean cells displayed in panel c, 3-60 deg in each hemisphere.")
print(f"  Opposite signs (CROSS-HATCHED)       : {sign_fractions['reversed']:.2f}%")
print(f"  Same nonzero signs            : {sign_fractions['same']:.2f}%")
print(f"  At least one exactly zero     : {sign_fractions['zero']:.2f}%")
print(f"  Not opposite (same + zero)    : {sign_fractions['not_reversed']:.2f}%")
print("Neither missing cells nor cyclic longitude duplicates enter the denominator.")
print("A large MLD-related trend difference (D < 0.5) does not by itself imply reversal.")
print("Suggested caption: Diagonal hatching in (a) and (b) indicates trends with "
      "P < 0.05. Cross-hatching in (c) marks opposite signs of wind-work trends "
      "between the evolving-MLD and climatological-MLD experiments, without a "
      "statistical significance requirement.")
print("=" * 68)
