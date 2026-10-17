# -*- coding: utf-8 -*-
"""Revised final Supplementary two-panel map: forcing-response trend correspondence.

Run from the PROJECT ROOT:
    python FigureS9.py

Uses only precomputed trends/p values. It does NOT rerun the slab model,
recompute trends, interpolate data, or change any Fig. 3 file.

Inputs:
    cache/Fig3_windstress_plot_data.npz
    cache/Fig3_climh_pvalue_1979_2025_v2.npz
    data/deeplearning_climmld_niw_power_input_1979_2025.nc
        (file metadata is checked only; the NetCDF data are NOT reloaded)

Three mutually exclusive states in the SAME fixed p(V_NI)<0.05 area:
    1 = both V_NI and wind-work significant; same slope sign
    2 = both significant; opposite slope sign
    3 = V_NI significant; wind-work trend NOT significant

Everything else is masked and shown as white (land is light grey).
Percentages are area-weighted using cos(latitude).

Outputs:
    plots/FigureS9.png
    plots/FigureS9.pdf
    plots/FigureS9_stats.csv
"""

from pathlib import Path
import csv
import numpy as np

ROOT = Path.cwd()
FIG3_CACHE = ROOT / "cache" / "Fig3_windstress_plot_data.npz"
CLIM_P_CACHE = ROOT / "cache" / "Fig3_climh_pvalue_1979_2025_v2.npz"
CLIM_SOURCE = ROOT / "data" / "deeplearning_climmld_niw_power_input_1979_2025.nc"
OUT = ROOT / "plots"

P_THRESHOLD = 0.05
MIN_ABS_LAT = 3.0
MAX_ABS_LAT = 60.0
CENTRAL_LONGITUDE = 180.0

FIG_WIDTH_MM = 183.0
FIG_HEIGHT_MM = 82.0
DPI = 400

# Palette and map styling aligned more closely with Fig. 3.
COLOR_SAME = "#6DC95D"
COLOR_OPPOSITE = "#D7301F"
COLOR_WW_NONSIG = "#F4D474"
COLOR_LAND = "#E0E0E0"
COAST_COLOR = "#222222"
GRID_COLOR = "#9A9A9A"

STATES = (
    (1, "Both significant, same sign", COLOR_SAME),
    (2, "Both significant, opposite sign", COLOR_OPPOSITE),
    (3, "Wind-work trend not significant", COLOR_WW_NONSIG),
)


def read_and_validate_inputs(root=ROOT):
    fig_path = root / 'cache' / FIG3_CACHE.name
    p_path = root / 'cache' / CLIM_P_CACHE.name
    src_path = root / 'data' / CLIM_SOURCE.name

    if not fig_path.is_file():
        raise FileNotFoundError(f'Missing Fig. 3 cache: {fig_path}')
    if not p_path.is_file():
        raise FileNotFoundError(
            f'Missing validated climatological-MLD p-values: {p_path}\n'
            'First run FigS_forcing_response_significance_diagnostic_v2.py.'
        )

    keys = ('lat', 'lon', 'slope_ni_dec', 'pval_ni',
            'slope_full_dec', 'pval_full', 'slope_climh_dec')
    with np.load(fig_path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise KeyError(f'Fig. 3 cache missing fields: {missing}')
        f = {k: np.asarray(z[k], dtype=np.float64) for k in keys}

    lat, lon = f['lat'], f['lon']
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError('Latitude and longitude must be 1-D arrays.')
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError('Latitude/longitude must be strictly increasing.')
    shape = (len(lat), len(lon))
    for k in keys[2:]:
        if f[k].shape != shape:
            raise ValueError(f'{k} has shape {f[k].shape}; expected {shape}')

    with np.load(p_path, allow_pickle=False) as z:
        needed = ('lat', 'lon', 'pval_climh', 'fig3_climh_slope', 'source_signature')
        absent = [k for k in needed if k not in z.files]
        if absent:
            raise ValueError(f'Climatological p-value cache missing: {absent}')
        if not (np.array_equal(z['lat'], lat) and np.array_equal(z['lon'], lon)):
            raise RuntimeError('The climatological p-value grid is different from Fig. 3.')
        if not np.array_equal(z['fig3_climh_slope'], f['slope_climh_dec'], equal_nan=True):
            raise RuntimeError('The p-value cache was produced for another Fig. 3 trend cache.')
        p_clim = np.asarray(z['pval_climh'], dtype=np.float64)
        source_signature = np.asarray(z['source_signature'], dtype=np.int64)

    if p_clim.shape != shape:
        raise ValueError(f'pval_climh has shape {p_clim.shape}; expected {shape}')
    if not src_path.is_file():
        raise FileNotFoundError(f'Missing original climatological source for cache check: {src_path}')
    stat = src_path.stat()
    now = np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)
    if not np.array_equal(source_signature, now):
        raise RuntimeError(
            'Climatological source metadata have changed since p-value caching.\n'
            'Please rerun FigS_forcing_response_significance_diagnostic_v2.py.'
        )
    f['pval_climh'] = p_clim
    return f


def classify(f):
    lat, lon = f['lat'], f['lon']
    v, ev, cm = f['slope_ni_dec'], f['slope_full_dec'], f['slope_climh_dec']
    pv, pe, pc = f['pval_ni'], f['pval_full'], f['pval_climh']

    lat_valid = (np.abs(lat) >= MIN_ABS_LAT) & (np.abs(lat) <= MAX_ABS_LAT)
    eligible = (
        lat_valid[:, None] & (pv < P_THRESHOLD)
        & np.isfinite(v) & np.isfinite(ev) & np.isfinite(cm)
        & np.isfinite(pv) & np.isfinite(pe) & np.isfinite(pc)
        & (v != 0) & (ev != 0) & (cm != 0)
    )
    if not eligible.any():
        raise ValueError('Common significant-V_NI analysis area is empty.')

    def one_state(wind_work_slope, wind_work_p):
        out = np.zeros(v.shape, dtype=np.uint8)
        ww_significant = wind_work_p < P_THRESHOLD
        same_sign = np.sign(v) == np.sign(wind_work_slope)
        out[eligible & ww_significant & same_sign] = 1
        out[eligible & ww_significant & ~same_sign] = 2
        out[eligible & ~ww_significant] = 3
        if not np.array_equal(out > 0, eligible):
            raise AssertionError('Three classes do not partition the fixed domain.')
        return out

    state_evolving = one_state(ev, pe)
    state_clim = one_state(cm, pc)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], v.shape)
    denominator = float(w[eligible].sum())

    def weighted_share(mask):
        return 100.0 * float(w[mask & eligible].sum()) / denominator

    fractions = {}
    for name, state in (('Evolving MLD', state_evolving),
                        ('Climatological MLD', state_clim)):
        fractions[name] = [weighted_share(state == code) for code in (1, 2, 3)]
        if not np.isclose(sum(fractions[name]), 100., atol=1e-9):
            raise AssertionError(f'Area partition failed: {name}')

    if not np.array_equal(state_evolving > 0, state_clim > 0):
        raise AssertionError('The two panels use different analysis areas.')
    return eligible, state_evolving, state_clim, fractions


def print_and_save_stats(eligible, states_e, states_c, fractions, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    print(f'\nCommon significant V_NI grid cells: {int(eligible.sum()):,}')
    print('All percentages are area-weighted within ONE fixed mask.')
    print('                              Evolving     Climatological')
    for idx, (_, label, _) in enumerate(STATES):
        a = fractions['Evolving MLD'][idx]
        b = fractions['Climatological MLD'][idx]
        print(f'{label:36s} {a:8.4f}%       {b:8.4f}%')
    print('Significant same-sign net change: '
          f'{fractions["Climatological MLD"][0]-fractions["Evolving MLD"][0]:+.4f} percentage points')

    expected = ((49.7303, 0.3879, 49.8817), (62.3549, 0.2222, 37.4229))
    for name, ref in zip(('Evolving MLD', 'Climatological MLD'), expected):
        if not np.allclose(fractions[name], ref, atol=.025, rtol=0):
            print(f'WARNING: {name} differs from previously confirmed results. '
                  'Check inputs and masks before submitting the figure.')

    csv_file = out / 'FigureS9_stats.csv'
    with csv_file.open('w', newline='', encoding='utf-8-sig') as h:
        wr = csv.writer(h)
        wr.writerow(['state', 'description', 'evolving_area_pct', 'climatological_area_pct',
                     'evolving_cells', 'climatological_cells'])
        for idx, (code, label, _) in enumerate(STATES):
            wr.writerow([code, label,
                         f'{fractions["Evolving MLD"][idx]:.8f}',
                         f'{fractions["Climatological MLD"][idx]:.8f}',
                         int(np.sum(states_e == code)), int(np.sum(states_c == code))])
    print(f'Saved: {csv_file}')


def draw_figure(f, state_evolving, state_clim, fractions, out=OUT):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.0,
        'axes.titlesize': 8.5,
        'axes.linewidth': 0.8,
        'savefig.dpi': DPI,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    palette = ListedColormap([c for _, _, c in STATES], name='three_states')
    palette.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], palette.N)

    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
                     facecolor='white')
    proj = ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)

    panel_defs = [
        {
            'pos': [0.030, 0.245, 0.455, 0.610],
            'state': state_evolving,
            'panel': 'a',
            'label': 'Evolving MLD',
            'stats_key': 'Evolving MLD',
        },
        {
            'pos': [0.515, 0.245, 0.455, 0.610],
            'state': state_clim,
            'panel': 'b',
            'label': 'Climatological MLD',
            'stats_key': 'Climatological MLD',
        },
    ]

    for item in panel_defs:
        ax = fig.add_axes(item['pos'], projection=proj)
        arr = item['state'].astype(float)
        arr[item['state'] == 0] = np.nan
        arr_cyclic, lon_cyclic = add_cyclic_point(arr, coord=f['lon'])

        ax.set_global()
        ax.pcolormesh(
            lon_cyclic, f['lat'], np.ma.masked_invalid(arr_cyclic),
            transform=ccrs.PlateCarree(), cmap=palette, norm=norm,
            shading='auto', rasterized=True, zorder=2,
        )
        ax.add_feature(cfeature.LAND, facecolor=COLOR_LAND, edgecolor='none', zorder=3)
        ax.coastlines(resolution='110m', color=COAST_COLOR, linewidth=0.52, zorder=5)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.35, color=GRID_COLOR,
                          alpha=0.42, linestyle='--', zorder=1, draw_labels=False)
        ax.set_title(item['panel'], loc='left', fontweight='bold', pad=4)

        # Small on-map case label, compact and unobtrusive.
        ax.text(0.035, 0.935, item['label'], transform=ax.transAxes,
                ha='left', va='top', fontsize=8.0,
                bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                          edgecolor='none', alpha=0.88),
                zorder=10)

        same_pct, opp_pct, _ = fractions[item['stats_key']]
        xmid = item['pos'][0] + item['pos'][2] / 2.0
        fig.text(xmid, 0.193,
                 f'Significant same-sign: {same_pct:.2f}%     Opposing: {opp_pct:.2f}%',
                 ha='center', va='center', fontsize=7.5)

    handles = [Patch(facecolor=color, edgecolor='none', label=label)
               for _, label, color in STATES]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.088),
               ncol=3, frameon=False, fontsize=7.4, columnspacing=1.8,
               handlelength=1.6, handleheight=0.95)

    out.mkdir(parents=True, exist_ok=True)
    png_file = out / 'FigureS9.png'
    pdf_file = out / 'FigureS9.pdf'
    fig.savefig(png_file, dpi=DPI, facecolor='white')
    fig.savefig(pdf_file, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'Saved: {png_file}')
    print(f'Saved: {pdf_file}')


def main():
    f = read_and_validate_inputs()
    eligible, state_e, state_c, fractions = classify(f)
    print_and_save_stats(eligible, state_e, state_c, fractions)
    draw_figure(f, state_e, state_c, fractions)


if __name__ == '__main__':
    main()
