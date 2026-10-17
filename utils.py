# utils.py
import random
import numpy as np
import torch
import config as C

def set_seed(seed=2025):
    """Fixes all random seeds for reproducible hardware execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def robust_mad_stats(X_samp, eps=1e-8):
    """Extracts non-parametric distribution statistics (Median Absolute Deviation)."""
    med = np.nanmedian(X_samp, axis=(0, 1))
    mad = np.nanmedian(np.abs(X_samp - med), axis=(0, 1)) + eps
    return med.astype(np.float32), (1.4826 * mad + eps).astype(np.float32)

def speed_mag_uv(u, v):
    """Computes vector magnitude strictly in non-negative domain."""
    return torch.sqrt(u * u + v * v + 1e-12)

@torch.no_grad()
def segment_rms(u, v):
    return torch.sqrt(torch.mean(u * u + v * v, dim=1) + 1e-12)

def forward_unpack(model, cond, slab):
    """Standardized API interface for architectural variations."""
    out = model(cond, slab)
    if len(out) == 4:
        return out[0], out[1], out[2], out[3], {}
    elif len(out) == 5:
        return out[0], out[1], out[2], out[4], {"log_r": out[3]}
    raise ValueError(f"Architectural output mismatch. Length: {len(out)}")

@torch.no_grad()
def eval_epoch(model, loader):
    """Standardized validation epoch evaluation."""
    model.eval()
    rs_list, rg_list, rp_list, logg_list = [], [], [], []

    for batch in loader:
        cond, slab, y = batch[0].to(C.DEVICE), batch[1].to(C.DEVICE), batch[2].to(C.DEVICE)
        yp, _, logg, _, _ = forward_unpack(model, cond, slab)

        rs_list.append(segment_rms(slab[:, 0, :], slab[:, 1, :]).cpu().numpy())
        rg_list.append(segment_rms(y[:, 0, :], y[:, 1, :]).cpu().numpy())
        rp_list.append(segment_rms(yp[:, 0, :], yp[:, 1, :]).cpu().numpy())
        logg_list.append(torch.mean(logg, dim=1).cpu().numpy())

    return (np.concatenate(rs_list) if rs_list else np.array([]),
            np.concatenate(rg_list) if rg_list else np.array([]),
            np.concatenate(rp_list) if rp_list else np.array([]),
            np.concatenate(logg_list) if logg_list else np.array([]))