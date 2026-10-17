# diagnostics.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import config as C
from utils import forward_unpack

def _binned_median_iqr(x, y, nbins=60, xlog=True, min_count=200):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return None

    if xlog:
        xmin, xmax = np.min(x[x > 0]), np.max(x)
        edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    else:
        edges = np.linspace(np.min(x), np.max(x), nbins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])
    med, q25, q75 = np.full(nbins, np.nan), np.full(nbins, np.nan), np.full(nbins, np.nan)
    cnt = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        sel = (x >= edges[i]) & (x < edges[i+1])
        c = int(np.sum(sel))
        cnt[i] = c
        if c < min_count: continue
        yy = y[sel]
        med[i] = np.median(yy)
        q25[i], q75[i] = np.percentile(yy, [25, 75])

    return centers, med, q25, q75, cnt

def save_D2_D3(outdir, rms_slab, rms_gt, rms_pred):
    os.makedirs(outdir, exist_ok=True)
    
    # D2: Absolute Log Error
    logerr_before = np.abs(np.log(np.maximum(rms_gt, 1e-12)) - np.log(np.maximum(rms_slab, 1e-12)))
    logerr_after  = np.abs(np.log(np.maximum(rms_gt, 1e-12)) - np.log(np.maximum(rms_pred, 1e-12)))

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    eb, ea = logerr_before[np.isfinite(logerr_before)], logerr_after[np.isfinite(logerr_after)]
    lo, hi = np.percentile(np.concatenate([eb, ea]), [0.5, 99.5])
    ax.hist(np.clip(eb, lo, hi), bins=C.DIAG_HIST_BINS, alpha=0.55, label="Before Correction")
    ax.hist(np.clip(ea, lo, hi), bins=C.DIAG_HIST_BINS, alpha=0.55, label="After Correction")
    ax.set_title("Absolute Log Error Distribution")
    ax.set_xlabel("|log(RMS_GT) - log(RMS_Pred)|")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D2_abslogerr_hist.png"), dpi=C.DIAG_DPI)
    plt.close(fig)

    # D2: Signed Log Error
    err_before = np.log(rms_slab + 1e-12) - np.log(rms_gt + 1e-12)
    err_after  = np.log(rms_pred + 1e-12) - np.log(rms_gt + 1e-12)
    
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    eb, ea = err_before[np.isfinite(err_before)], err_after[np.isfinite(err_after)]
    lim = np.percentile(np.abs(np.concatenate([eb, ea])), 99.5)

    ax.hist(np.clip(eb, -lim, lim), bins=C.DIAG_HIST_BINS, alpha=0.55, label="Before Correction")
    ax.hist(np.clip(ea, -lim, lim), bins=C.DIAG_HIST_BINS, alpha=0.55, label="After Correction")
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_title("Signed Log Error Distribution")
    ax.set_xlabel("log(RMS_Pred) - log(RMS_GT)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D2_signed_logerr_hist.png"), dpi=C.DIAG_DPI)
    plt.close(fig)

    # D3: Binned Median Ratio
    ratio_before = rms_gt / (rms_slab + 1e-12)
    ratio_after  = rms_gt / (rms_pred + 1e-12)
    res1 = _binned_median_iqr(rms_slab, ratio_before, nbins=C.DIAG_NBINS, xlog=C.DIAG_XLOG_BINS, min_count=C.DIAG_BIN_MINCNT)
    res2 = _binned_median_iqr(rms_slab, ratio_after,  nbins=C.DIAG_NBINS, xlog=C.DIAG_XLOG_BINS, min_count=C.DIAG_BIN_MINCNT)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.4), sharex=True)
    if res1 is not None:
        x1, m1, q1l, q1h, c1 = res1
        ok = np.isfinite(m1)
        axes[0].plot(x1[ok], m1[ok], label="Before: Median(GT/Slab)")
        axes[0].fill_between(x1[ok], q1l[ok], q1h[ok], alpha=0.2)
        axes[1].plot(x1, np.log10(np.maximum(c1, 1)), label="Count Before")
    if res2 is not None:
        x2, m2, q2l, q2h, c2 = res2
        ok = np.isfinite(m2)
        axes[0].plot(x2[ok], m2[ok], label="After: Median(GT/Pred)")
        axes[0].fill_between(x2[ok], q2l[ok], q2h[ok], alpha=0.2)
        axes[1].plot(x2, np.log10(np.maximum(c2, 1)), label="Count After")

    axes[0].axhline(1.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Amplitude Ratio (GT / Model)")
    axes[0].legend()
    axes[1].set_ylabel("log10(Count per Bin)")
    axes[1].set_xlabel("RMS Slab (m/s)")
    axes[1].legend()
    
    if C.DIAG_XLOG_BINS:
        axes[1].set_xscale("log")
        axes[1].set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
        axes[1].get_xaxis().set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        
    fig.savefig(os.path.join(outdir, "D3_binned_median_ratio_vs_rms_slab.png"), dpi=C.DIAG_DPI)
    plt.close(fig)

def plot_random_timeseries(model, ds, outdir, nplot=10, seed=0):
    """Hidden utility for verifying local reproducibility."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ds), size=min(nplot, len(ds)), replace=False)
    model.eval()

    for k, ii in enumerate(pick):
        item = ds[ii]
        cond, slab, y = item[0].unsqueeze(0).to(C.DEVICE), item[1].unsqueeze(0).to(C.DEVICE), item[2].unsqueeze(0).to(C.DEVICE)

        with torch.no_grad():
            yp, _, _, _, _ = forward_unpack(model, cond, slab)

        ugt, vgt = y[0, 0].cpu().numpy(), y[0, 1].cpu().numpy()
        us, vs = slab[0, 0].cpu().numpy(), slab[0, 1].cpu().numpy()
        up, vp = yp[0, 0].cpu().numpy(), yp[0, 1].cpu().numpy()
        t = np.arange(ugt.size)

        fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
        axes[0].plot(t, np.sqrt(ugt**2 + vgt**2), label="|GT|", lw=1.6)
        axes[0].plot(t, np.sqrt(us**2 + vs**2), label="|Slab|", lw=1.0, alpha=0.9)
        axes[0].plot(t, np.sqrt(up**2 + vp**2), label="|Pred|", lw=1.0, alpha=0.9)
        axes[0].set_ylabel("Speed (m/s)"); axes[0].legend(loc="upper right")
        
        axes[1].plot(t, ugt, label="u_GT", lw=1.3)
        axes[1].plot(t, us, label="u_Slab", lw=1.0, alpha=0.8)
        axes[1].plot(t, up, label="u_Pred", lw=1.0, alpha=0.8)
        axes[1].set_ylabel("u (m/s)"); axes[1].legend(loc="upper right")
        
        axes[2].plot(t, vgt, label="v_GT", lw=1.3)
        axes[2].plot(t, vs, label="v_Slab", lw=1.0, alpha=0.8)
        axes[2].plot(t, vp, label="v_Pred", lw=1.0, alpha=0.8)
        axes[2].set_ylabel("v (m/s)"); axes[2].set_xlabel("Time Step")
        axes[2].legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"TS_{k:02d}.png"), dpi=220)
        plt.close(fig)