# model.py
import torch
import torch.nn as nn
import config as C
from utils import speed_mag_uv

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, d=1, p=None):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, dilation=d, padding=p)
        self.gn   = nn.GroupNorm(num_groups=min(8, cout), num_channels=cout)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class DilatedTCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation."""
    def __init__(self, c, layers=7):
        super().__init__()
        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks.append(ConvBlock(c, c, k=3, d=d))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class AmpGainResNet(nn.Module):
    """
    Physics-guided Amplitude Gain and Residual Network.
    
    Inputs:
      cond: [B, 8, T] - Environmental forcing and geospatial conditioning.
      slab: [B, 2, T] - Baseline physical slab model predictions.
      
    Outputs:
      y_pred: [B, 2, T] - Final synthesized velocity components.
      s_hat : [B, T]    - Synthesized speed magnitude.
      log_g : [B, T]    - Kinematic scaling factor (log domain).
      log_r : [B, T]    - Residual amplitude correction (log domain).
      s_slab: [B, T]    - Original slab speed magnitude.
    """
    def __init__(self, c_in=8, c=128,
                 log_g_clamp=(-2.5, 2.5),  # Scaling bounds (approx. g in [0.08, 12.2])
                 log_r_clamp=(-9.0, 0.45)): # Residual bounds (approx. r in [0.0, 1.57] for physical limits)
        super().__init__()
        self.log_g_clamp = log_g_clamp
        self.log_r_clamp = log_r_clamp

        self.stem = nn.Sequential(
            ConvBlock(c_in, c, k=5),
            ConvBlock(c, c, k=3),
        )
        self.tcn = DilatedTCN(c, layers=7)

        # Dual-head prediction for dynamic gain and residual correction
        self.head = nn.Sequential(
            ConvBlock(c, c, k=3),
            nn.Conv1d(c, 2, kernel_size=1),
        )

    def forward(self, cond, slab):
        feat = self.tcn(self.stem(cond))        # [B, c, T]
        out  = self.head(feat)                  # [B, 2, T]
        log_g = out[:, 0, :]                    # [B, T]
        log_r = out[:, 1, :]                    # [B, T]

        # Numerical stability and physical limit clamping
        lg0, lg1 = self.log_g_clamp
        lr0, lr1 = self.log_r_clamp
        log_g = torch.clamp(log_g, lg0, lg1)
        log_r = torch.clamp(log_r, lr0, lr1)

        g = torch.exp(log_g)                    # [B, T] >= 0
        r = torch.exp(log_r)                    # [B, T] >= 0

        u_s = slab[:, 0, :]
        v_s = slab[:, 1, :]
        s_slab = speed_mag_uv(u_s, v_s)         # [B, T]
        
        # Extract normalized directional components
        ux = u_s / (s_slab + C.EPS)
        vx = v_s / (s_slab + C.EPS)

        # Core physical synthesis equation: Prediction = Gain * Slab_Physics + Data_Driven_Residual
        s_hat = g * s_slab + r                  # [B, T]

        # Re-apply synthesized amplitude to the kinematic direction
        u_p = s_hat * ux
        v_p = s_hat * vx
        y_pred = torch.stack([u_p, v_p], dim=1) # [B, 2, T]

        return y_pred, s_hat, log_g, log_r, s_slab