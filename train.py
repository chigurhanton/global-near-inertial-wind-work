import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from utils import set_seed, robust_mad_stats, speed_mag_uv, eval_epoch
from dataset import DrifterDataset
from model import AmpGainResNet
from losses import build_loss_scaler_from_train, amp_gain_res_loss, smooth_envelope_proxy
from diagnostics import save_D2_D3

def parse_args():
    parser = argparse.ArgumentParser(description="Physics-guided deep learning for oceanic near-inertial waves.")
    parser.add_argument("--data_path", type=str, default=r"data/drifter_niw_300h_new_oras5_mld_with_raw.npz", help="Path to the input NPZ dataset.")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save weights and logs.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")
    return parser.parse_args()

def calculate_qc_thresholds(X, Y, idx_tr, rng, nsamp_seg=60000):
    """
    Computes QC thresholds using the coherence between the slab baseline and observations.
    """
    pick = rng.choice(idx_tr, size=min(nsamp_seg, idx_tr.size), replace=False)
    Xp = X[pick].astype(np.float32)
    Yp = Y[pick].astype(np.float32)

    # 1. Wind stress threshold (tau0) for low-wind masking
    tx = Xp[:, :, C.IDX_TX]
    ty = Xp[:, :, C.IDX_TY]
    tau = np.sqrt(tx**2 + ty**2)

    w = getattr(C, "CLEAN_ENV_WIN", 16)
    if w > 1:
        pad = w - 1
        pl = pad // 2
        pr = pad - pl
        tau_pad = np.pad(tau, ((0, 0), (pl, pr)), mode="edge")
        ker = np.ones(w, dtype=np.float32) / w
        tau_env = np.apply_along_axis(lambda a: np.convolve(a, ker, mode="valid"), 1, tau_pad)
    else:
        tau_env = tau

    tau_env = tau_env.reshape(-1)
    tau_env = tau_env[np.isfinite(tau_env)]
    q0 = float(getattr(C, "CLEAN_TAU_Q0", 0.20))
    tau0 = float(np.quantile(tau_env, q0)) if tau_env.size > 0 else 0.05
    tau0 = max(tau0, 1e-6)

    # 2. Consistency ratio threshold (r_hi) using s_gt vs s_slab
    # Identifies instances where observed s_gt is unphysically larger than s_slab capacity
    u_sl, v_sl = Xp[:, :, C.IDX_U_SLAB], Xp[:, :, C.IDX_V_SLAB]
    u_gt, v_gt = Yp[:, :, C.IDX_U_GT], Yp[:, :, C.IDX_V_GT]
    
    s_sl = np.sqrt(u_sl**2 + v_sl**2).reshape(-1)
    s_gt = np.sqrt(u_gt**2 + v_gt**2).reshape(-1)

    # Filter for valid regimes: s_sl must be non-negligible to evaluate the ratio safely
    m = (s_sl > 1e-3) & (s_gt > 0.05) 
    ratio = s_gt[m] / (s_sl[m] + 1e-12)
    
    qhi = float(getattr(C, "CLEAN_RATIO_QHI", 0.995))
    if ratio.size > 0:
        r_hi = float(np.quantile(ratio, qhi))
        r_hi = max(r_hi, 5.0) # Ensure at least a 5x correction allowance
    else:
        r_hi = 8.0

    return tau0, r_hi

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    D = np.load(args.data_path, mmap_mode="r")
    X, Y = D["X"], D["Y"]
    N, _, _ = X.shape

    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    ntr = int(C.TRAIN_FRAC * N)
    idx_tr, idx_va = idx[:ntr], idx[ntr:]

    # Calculate QC Thresholds using the physics-consistency logic
    tau0, r_hi = calculate_qc_thresholds(X, Y, idx_tr, rng)
    print(f"[QC Init] tau0={tau0:.5f}, consistency_r_hi={r_hi:.2f}")

    cond_mean, cond_std = None, None
    if C.NORMALIZE_COND:
        nsamp = min(120000, idx_tr.size)
        samp = rng.choice(idx_tr, size=nsamp, replace=False)
        Xs = X[samp][:, :, C.COND_COLS].astype(np.float32)
        cond_mean, cond_std = robust_mad_stats(Xs)
        cond_mean, cond_std = cond_mean.reshape(1, -1), cond_std.reshape(1, -1)

    ds_tr = DrifterDataset(args.data_path, idx_tr, cond_mean, cond_std, C.NORMALIZE_COND)
    ds_va = DrifterDataset(args.data_path, idx_va, cond_mean, cond_std, C.NORMALIZE_COND)
    
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = AmpGainResNet(c_in=len(C.COND_COLS), c=128).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    
    use_amp = C.AMP and C.DEVICE.startswith("cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    loss_cfg = build_loss_scaler_from_train(model, dl_tr, device=C.DEVICE)
    loss_cfg.update({"alpha_r": 0.2, "beta_g": 0.005})

    tau_floor = float(getattr(C, "CLEAN_TAU_FLOOR", 0.05))
    ratio_sharp = float(getattr(C, "CLEAN_RATIO_SHARP", 10.0))
    env_win = int(getattr(C, "CLEAN_ENV_WIN", 16))
    
    # Speed threshold to activate consistency check (avoids masking background noise)
    s_ref = 0.05 

    best_val_loss = float("inf")
    patience_counter = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, nstep = 0.0, 0

        for cond, slab, y, _ in dl_tr:
            cond = cond.to(C.DEVICE, non_blocking=True)
            slab = slab.to(C.DEVICE, non_blocking=True)
            y = y.to(C.DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # 1. Background wind stress mask (m_tau)
            tx, ty = cond[:, 2, :], cond[:, 3, :]
            tau = torch.sqrt(tx**2 + ty**2 + 1e-12)
            tau_env = smooth_envelope_proxy(tau, win=env_win)
            m_tau = torch.clamp((tau_env / (tau_env + tau0)).detach(), min=tau_floor, max=1.0)

            # 2. Forward pass
            with torch.autocast(device_type='cuda', enabled=use_amp):
                yp, s_hat, log_g, log_r, s_slab = model(cond, slab)
                
                # 3. Consistency Mask: Identify non-local energy (Strong GT + Weak Slab)
                s_gt = speed_mag_uv(y[:, 0, :], y[:, 1, :])
                ratio = (s_gt / (s_slab + 1e-6)).detach()
                
                # Only apply the ratio mask if the observed current is physically significant
                ratio = torch.where(s_gt > s_ref, ratio, torch.zeros_like(ratio))
                m_ratio = torch.sigmoid(-ratio_sharp * (ratio - r_hi))
                
                # Final physical gating mask
                qc_mask = m_tau * m_ratio

                # 4. Loss computation
                r = torch.exp(log_r)
                L, _ = amp_gain_res_loss(s_hat, s_gt, s_slab, log_g, r, loss_cfg, m=qc_mask)

            # 5. Backpropagation
            if scaler is not None:
                scaler.scale(L).backward()
                scaler.step(opt)
                scaler.update()
            else:
                L.backward()
                opt.step()

            loss_sum += float(L.item())
            nstep += 1

        tr_loss = loss_sum / max(1, nstep)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            vloss_sum, vn = 0.0, 0
            for cond, slab, y, _ in dl_va:
                cond = cond.to(C.DEVICE, non_blocking=True)
                slab = slab.to(C.DEVICE, non_blocking=True)
                y = y.to(C.DEVICE, non_blocking=True)
                
                # Replicate mask logic for validation tracking
                tx, ty = cond[:, 2, :], cond[:, 3, :]
                tau = torch.sqrt(tx**2 + ty**2 + 1e-12)
                tau_env = smooth_envelope_proxy(tau, win=env_win)
                m_tau = torch.clamp((tau_env / (tau_env + tau0)).detach(), min=tau_floor, max=1.0)

                with torch.autocast(device_type='cuda', enabled=use_amp):
                    yp, s_hat, log_g, log_r, s_slab = model(cond, slab)
                    
                    s_gt = speed_mag_uv(y[:, 0, :], y[:, 1, :])
                    ratio = (s_gt / (s_slab + 1e-6)).detach()
                    ratio = torch.where(s_gt > s_ref, ratio, torch.zeros_like(ratio))
                    m_ratio = torch.sigmoid(-ratio_sharp * (ratio - r_hi))
                    qc_mask = m_tau * m_ratio
                    
                    r = torch.exp(log_r)
                    L, _ = amp_gain_res_loss(s_hat, s_gt, s_slab, log_g, r, loss_cfg, m=qc_mask)

                vloss_sum += float(L.item())
                vn += 1
            va_loss = vloss_sum / max(1, vn)

        print(f"[Epoch {ep:03d}] Train: {tr_loss:.5f} | Val: {va_loss:.5f} | Time: {time.time()-t0:.1f}s")

        # Early Stopping Check
        if ep >= C.EARLY_STOP_MIN_EPOCHS:
            if va_loss < best_val_loss - C.EARLY_STOP_MIN_DELTA:
                best_val_loss = va_loss
                patience_counter = 0
                ckpt = {
                    "model": model.state_dict(), 
                    "opt": opt.state_dict(), 
                    "cond_mean": cond_mean, 
                    "cond_std": cond_std
                }
                torch.save(ckpt, os.path.join(args.out_dir, "best_amp_gain_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= C.EARLY_STOP_PATIENCE:
                    print(f"Early stopping triggered at epoch {ep}.")
                    break

if __name__ == "__main__":
    main()