"""
One-click runner for the global inference script.

Usage:
    python run_global_infer_gainres_stripe_fast2_timed.py
"""

import os
import sys
import subprocess

def main():
    cfg = dict(
        era_dir   = r"E:\era5_wind_data_1979to2025",
        slab_dir  = r"H:\inertial_with_mld",
        out_dir   = r"H:\inertial_pred_by_model",   # Recommendation: Use an internal SSD for output to optimize I/O performance
        ckpt      = r"F:\Python_Code\Python_vscode\niw_project_20160112_1720_for_QCdata\outputs_stage1_cleaned\best_amp_gain_model.pt",
        mld_dir   = r"H:\oras5_mld_1980_2022\mld_monthly",  # Directory containing oras5_mld_YYYYMM.mat files
        mld_pat   = "oras5_mld_{yy:04d}{mm:02d}.mat",

        # Optional staging directory for I/O optimization (set to None if no performance gain is observed)
        staging_dir = None, #r"F:\_tmp_slab",

        # Single month execution
        # year  = 2019,
        # month = 12,

        # Batch processing (comment out year/month above and use year_start/year_end)
        year_start = 1979,
        year_end   = 2025,

        # Key parameters for spatial striping and memory management
        stripe_lon  = 360,     # Longitudinal stripe width (e.g., 64/96/128); larger values increase memory and I/O load
        tile_lat    = 120,     # Latitudinal tile height (e.g., 32/48/60); 48 is recommended for 8GB VRAM
        point_batch = 128,     # Batch size for point-wise inference; decrease if Out-Of-Memory (OOM) errors occur

        device = "cuda",
    )

    # Target script located in the same directory
    script = os.path.join(os.path.dirname(__file__), "global_pred_niw.py")

    cmd = [
        sys.executable, script,
        "--era_dir", cfg["era_dir"],
        "--slab_dir", cfg["slab_dir"],
        "--out_dir",  cfg["out_dir"],
        "--ckpt",     cfg["ckpt"],
        "--mld_dir", cfg["mld_dir"],
        "--mld_pat", cfg["mld_pat"],
        "--stripe_lon", str(cfg["stripe_lon"]),
        "--tile_lat", str(cfg["tile_lat"]),
        "--point_batch", str(cfg["point_batch"]),
        "--device", cfg["device"],
    ]

    # Append optional staging directory
    if cfg.get("staging_dir", None):
        cmd += ["--staging_dir", cfg["staging_dir"]]

    # Append time constraints (single month vs. multi-year batch)
    if cfg.get("year", None) is not None and cfg.get("month", None) is not None:
        cmd += ["--year", str(cfg["year"]), "--month", str(cfg["month"])]
    else:
        cmd += ["--year_start", str(cfg["year_start"]), "--year_end", str(cfg["year_end"])]

    print("\n[RUN CMD]\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()