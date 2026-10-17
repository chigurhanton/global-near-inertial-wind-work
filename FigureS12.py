# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S10.

Binned ratio correction diagnostics comparing the baseline slab model and the
physics-informed deep-learning correction.

Designed for execution as a single cell in a VS Code Jupyter notebook.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = PROJECT_DIR / "cache"
PLOTS_DIR = PROJECT_DIR / "plots"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH = DATA_DIR / "drifter_niw_300h_new_oras5_mld_with_raw.npz"
CKPT_PATH = OUTPUTS_DIR / "best_amp_gain_model.pt"
CACHE_FILE = CACHE_DIR / "FigS12_plot_data.npz"
FIGURE_FILE = PLOTS_DIR / "FigureS12.png"

BATCH_SIZE = 256
TRAIN_FRACTION = 0.8
RNG_SEED = 2025
CACHE_VERSION = 1
RECOMPUTE_CACHE = False


# =============================================================================
# 2. SCIENTIFIC SETTINGS (kept consistent with the original code)
# =============================================================================
MIN_COUNT = 50
NBINS = 60


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
# 4. CORE UTILS
# =============================================================================
def compute_rms_series():
    """Run the original validation inference and cache rs / rg / rp."""
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {NPZ_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {CKPT_PATH}")

    import torch
    from torch.utils.data import DataLoader

    import config as C
    from model import AmpGainResNet
    from dataset import DrifterDataset
    from utils import forward_unpack

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    cond_mean = checkpoint.get("cond_mean", None)
    cond_std = checkpoint.get("cond_std", None)

    data = np.load(NPZ_PATH, mmap_mode="r")
    n_samples = data["X"].shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(indices)
    n_train = int(TRAIN_FRACTION * n_samples)
    idx_val = indices[n_train:]

    ds_val = DrifterDataset(
        str(NPZ_PATH),
        idx_val,
        cond_mean=cond_mean,
        cond_std=cond_std,
        normalize_cond=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = AmpGainResNet(c_in=8, c=128).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rms_slab_list, rms_gt_list, rms_pred_list = [], [], []

    with torch.no_grad():
        for batch in dl_val:
            cond, slab, y = batch[0], batch[1], batch[2]
            cond = cond.to(device)
            slab = slab.to(device)
            y = y.to(device)

            yp, s_hat, log_g, s_slab, extra = forward_unpack(model, cond, slab)

            r_s = torch.sqrt(torch.mean(slab[:, 0, :] ** 2 + slab[:, 1, :] ** 2, dim=1))
            r_g = torch.sqrt(torch.mean(y[:, 0, :] ** 2 + y[:, 1, :] ** 2, dim=1))
            r_p = torch.sqrt(torch.mean(yp[:, 0, :] ** 2 + yp[:, 1, :] ** 2, dim=1))

            rms_slab_list.append(r_s.cpu().numpy())
            rms_gt_list.append(r_g.cpu().numpy())
            rms_pred_list.append(r_p.cpu().numpy())

    rs = np.concatenate(rms_slab_list)
    rg = np.concatenate(rms_gt_list)
    rp = np.concatenate(rms_pred_list)

    valid = np.isfinite(rs) & np.isfinite(rg) & np.isfinite(rp)
    rs, rg, rp = rs[valid], rg[valid], rp[valid]

    np.savez_compressed(
        CACHE_FILE,
        cache_version=np.array(CACHE_VERSION, dtype=np.int16),
        rng_seed=np.array(RNG_SEED, dtype=np.int16),
        train_fraction=np.array(TRAIN_FRACTION, dtype=float),
        batch_size=np.array(BATCH_SIZE, dtype=np.int16),
        rs=rs,
        rg=rg,
        rp=rp,
    )


def load_or_build_cache():
    """Load cached inference outputs or build them once."""
    if CACHE_FILE.exists() and not RECOMPUTE_CACHE:
        with np.load(CACHE_FILE, allow_pickle=False) as cache:
            required = {"cache_version", "rs", "rg", "rp"}
            if required.issubset(cache.files) and int(cache["cache_version"]) == CACHE_VERSION:
                return np.array(cache["rs"]), np.array(cache["rg"]), np.array(cache["rp"]), "cache"

    compute_rms_series()
    with np.load(CACHE_FILE, allow_pickle=False) as cache:
        return np.array(cache["rs"]), np.array(cache["rg"]), np.array(cache["rp"]), "source"


# =============================================================================
# 5. BINNED STATISTICS
# =============================================================================
def binned_stats_logx(x_data, y_data, nbins=NBINS, min_count=MIN_COUNT):
    """Median and IQR of y in logarithmically spaced x bins."""
    xmin = np.min(x_data[x_data > 0])
    xmax = np.max(x_data)
    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

    x_mid, med, q25, q75 = [], [], [], []
    for i in range(nbins):
        mask = (x_data >= edges[i]) & (x_data < edges[i + 1])
        if np.sum(mask) < min_count:
            continue
        x_bin = x_data[mask]
        y_bin = y_data[mask]
        x_mid.append(np.median(x_bin))
        med.append(np.median(y_bin))
        q25.append(np.percentile(y_bin, 25))
        q75.append(np.percentile(y_bin, 75))

    return np.array(x_mid), np.array(med), np.array(q25), np.array(q75)


# =============================================================================
# 6. LOAD DATA AND COMPUTE CURVES
# =============================================================================
rs, rg, rp, data_source = load_or_build_cache()

ratio_before = rg / (rs + 1e-12)
ratio_after = rg / (rp + 1e-12)

x1, m1, l1, h1 = binned_stats_logx(rs, ratio_before)
x2, m2, l2, h2 = binned_stats_logx(rs, ratio_after)


# =============================================================================
# 7. PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 4.3), facecolor="white")
fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.97)

ax.grid(True, which="major", linestyle="--", color="gray", alpha=0.4, zorder=0)
ax.grid(True, which="minor", linestyle=":", color="lightgray", alpha=0.3, zorder=0)

stroke_white = [pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()]

ax.fill_between(x1, l1, h1, color="tab:blue", alpha=0.20, zorder=2)
ax.plot(
    x1,
    m1,
    color="tab:blue",
    linewidth=2.1,
    path_effects=stroke_white,
    zorder=10,
    label="Baseline slab model",
)

ax.fill_between(x2, l2, h2, color="tab:orange", alpha=0.20, zorder=3)
ax.plot(
    x2,
    m2,
    color="tab:orange",
    linewidth=2.1,
    path_effects=stroke_white,
    zorder=11,
    label="Physics-informed DL correction",
)

ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, alpha=0.85, zorder=5)

ax.set_xscale("log")
ax.set_xlabel("Segment RMS baseline slab speed (m s$^{-1}$)")
ax.set_ylabel("Segment RMS amplitude ratio\n(ground truth / model)")
ax.set_xticks([0.01, 0.05, 0.1, 0.2, 0.4])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis="both", width=0.8, length=3)

ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.2)

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
