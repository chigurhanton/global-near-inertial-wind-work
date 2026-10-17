# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import config as C

class DrifterDataset(Dataset):
    """
    Standardized dataloader providing structural conditions, physical slab dynamics, 
    and ground truth observation constraints.
    """
    def __init__(self, npz_path, indices, cond_mean=None, cond_std=None, normalize_cond=True):
        self.npz = np.load(npz_path, mmap_mode="r")
        self.X = self.npz["X"]
        self.Y = self.npz["Y"]
        self.idx = indices.astype(np.int64)

        self.normalize_cond = normalize_cond
        self.cond_mean = cond_mean
        self.cond_std  = cond_std

    def __len__(self):
        return self.idx.size

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j].astype(np.float32)
        y = self.Y[j].astype(np.float32)

        slab = x[:, [C.IDX_U_SLAB, C.IDX_V_SLAB]].copy()
        slab[:, 0] = np.clip(slab[:, 0], -C.SLAB_CLIP, C.SLAB_CLIP)
        slab[:, 1] = np.clip(slab[:, 1], -C.SLAB_CLIP, C.SLAB_CLIP)

        cond = x[:, C.COND_COLS].copy()
        if self.normalize_cond and (self.cond_mean is not None) and (self.cond_std is not None):
            cond = (cond - self.cond_mean) / self.cond_std

        return (
            torch.from_numpy(cond).transpose(0, 1),
            torch.from_numpy(slab).transpose(0, 1),
            torch.from_numpy(y).transpose(0, 1),
            j
        )