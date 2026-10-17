# config.py
import torch

# =========================
# Feature Indices (T=300, Cx=12)
# Features: ["dtx", "dty", "tx", "ty", "H", "lon", "lat", "f", "u_slab", "v_slab", "u_raw", "v_raw"]
# =========================
IDX_DTX, IDX_DTY = 0, 1
IDX_TX,  IDX_TY  = 2, 3
IDX_H            = 4
IDX_LON, IDX_LAT = 5, 6
IDX_F            = 7
IDX_U_SLAB, IDX_V_SLAB = 8, 9
IDX_U_RAW,  IDX_V_RAW  = 10, 11

# Ground Truth Indices: [u_gt, v_gt]
IDX_U_GT, IDX_V_GT = 0, 1

# Stage-1 conditioning features (8 channels)
COND_COLS = [IDX_DTX, IDX_DTY, IDX_TX, IDX_TY, IDX_H, IDX_LON, IDX_LAT, IDX_F]

# =========================
# Training Setup
# =========================
TRAIN_FRAC = 0.8
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True

# Early Stopping Configuration
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_MIN_EPOCHS = 3

# Data Preprocessing
NORMALIZE_COND = True
SLAB_CLIP = 2.0
EPS = 1e-6

# =========================
# Loss Function Weights
# =========================
W_LOGSPEED = 1.0
W_SPEED    = 0.5
W_ENV      = 1.0
W_SMOOTH   = 0.05
ENV_WIN    = 24

# =========================
# Quality Control (QC) & Masking Hyperparameters
# =========================
CLEAN_ENV_WIN = 16       # Smoothing window for background wind stress (tau)
CLEAN_TAU_Q0  = 0.30     # Quantile threshold for valid wind stress environments
CLEAN_TAU_FLOOR = 0.05   # Minimum weight floor for low-wind segments
CLEAN_RATIO_QHI = 0.995  # Upper quantile bound for GT/Raw anomaly filtering
CLEAN_RATIO_SHARP = 10.0 # Steepness of the sigmoid decay for anomalous points

# =========================
# Diagnostics & Plotting
# =========================
PLOT_EVERY = 1
DIAG_HIST_BINS = 200
DIAG_NBINS = 60
DIAG_BIN_MINCNT = 100
DIAG_XLOG_BINS = True
DIAG_DPI = 300