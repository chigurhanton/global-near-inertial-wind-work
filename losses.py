# losses.py
import numpy as np
import torch
import torch.nn.functional as F
import config as C
from utils import speed_mag_uv

def smooth_envelope_proxy(s: torch.Tensor, win: int = 24) -> torch.Tensor:
    """Computes a smoothed envelope proxy with zero phase shift padding."""
    if win <= 1:
        return s
    x = s.unsqueeze(1)
    pad_total = win - 1
    pad_left = pad_total // 2
    x = F.pad(x, (pad_left, pad_total - pad_left), mode="replicate")
    return F.avg_pool1d(x, kernel_size=win, stride=1).squeeze(1)

def huber_pt(err: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    """Point-wise robust Huber penalty."""
    abs_e = torch.abs(err)
    d = torch.tensor(delta, device=err.device, dtype=err.dtype)
    quad = torch.minimum(abs_e, d)
    return 0.5 * quad * quad / d + (abs_e - quad)

def weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (x * w).sum() / (w.sum() + eps)

def w_slab_time(s_slab: torch.Tensor, s0: float = 0.02, floor: float = 0.25) -> torch.Tensor:
    """Downweights regions with negligible physical slab amplitude."""
    return torch.clamp((s_slab / (s_slab + s0)).detach(), min=floor, max=1.0)

def w_gt_tail(s_gt: torch.Tensor, s_ref: float, s_hi: float, gamma: float = 0.7, wmax: float = 6.0) -> torch.Tensor:
    """Quantile-adaptive weighting to prevent systematic underestimation of high-energy events."""
    s_gt_clip = torch.clamp(s_gt, 0.0, s_hi)
    w = (s_gt_clip / s_ref) ** gamma
    return torch.clamp(w, min=1.0, max=wmax).detach()

@torch.no_grad()
def build_loss_scaler_from_train(model, dl_tr, max_batches=80, device="cuda", q_ref=0.85, gamma=1.0, wmax=10.0, eps=1e-6):
    """Extracts empirical distribution statistics from the training set to adaptively scale loss components."""
    s_all = []
    for n, batch in enumerate(dl_tr):
        y = batch[2].to(device)
        s_all.append(speed_mag_uv(y[:,0,:], y[:,1,:]).reshape(-1).cpu().numpy())
        if n >= max_batches: break

    s_all = np.concatenate(s_all)
    s_all = s_all[np.isfinite(s_all) & (s_all > 0)]
    s_ref = float(np.quantile(s_all, q_ref)) if s_all.size > 0 else 0.05

    return {"s_ref": s_ref, "q_ref": q_ref, "gamma": gamma, "wmax": wmax, "eps": eps}

def amp_gain_res_loss(s_hat, s_gt, s_slab, log_g, r, loss_cfg, m=None):
    """
    Computes the composite objective function integrating robust envelope tracking, 
    kinematic smoothing, and structural regularization.
    """
    eps = C.EPS
    wsl = w_slab_time(s_slab, s0=loss_cfg.get("s0_slab", 0.02), floor=loss_cfg.get("w_slab_floor", 0.25))
    wgt = w_gt_tail(s_gt, loss_cfg.get("s_ref", 0.2), loss_cfg.get("s_hi", 0.4), loss_cfg.get("gamma", 0.7), loss_cfg.get("w_gt_max", 6.0))
    w = wsl * wgt if m is None else wsl * wgt * m

    log_s_hat, log_s_gt = torch.log(s_hat + eps), torch.log(s_gt + eps)
    
    L_log = weighted_mean(huber_pt(log_s_hat - log_s_gt, delta=loss_cfg.get("huber_delta_log", 0.25)), w)
    L_spd = weighted_mean(huber_pt(s_hat - s_gt, delta=loss_cfg.get("huber_delta_spd", 0.15)), w)
    
    L_env = weighted_mean(huber_pt(
        smooth_envelope_proxy(s_hat, C.ENV_WIN) - smooth_envelope_proxy(s_gt, C.ENV_WIN), 
        delta=loss_cfg.get("huber_delta_env", 0.15)), w)

    d = log_s_hat[:, 1:] - log_s_hat[:, :-1]
    L_smooth = weighted_mean(torch.abs(d), 0.5 * (w[:, 1:] + w[:, :-1]))

    L_r = weighted_mean(r, wsl)
    L_g = weighted_mean(torch.abs(log_g), wsl)

    L = (C.W_LOGSPEED * L_log + C.W_SPEED * L_spd + C.W_ENV * L_env + C.W_SMOOTH * L_smooth + 
         loss_cfg.get("alpha_r", 0.02) * L_r + loss_cfg.get("beta_g", 0.005) * L_g)

    return L, {}