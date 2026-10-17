import json
import numpy as np
from pathlib import Path

# --------- 读 v7.3 mat 的推荐方式：h5py ----------
# v7.3 MAT 本质是 HDF5
import h5py


def load_mat_var(mat_path: str, var_name: str) -> np.ndarray:
    """
    Load a variable from MATLAB v7.3 .mat file using h5py.
    Returns a numpy array with MATLAB column-major order preserved,
    then we transpose if needed later.
    """
    mat_path = str(mat_path)
    with h5py.File(mat_path, "r") as f:
        if var_name not in f:
            raise KeyError(f"Variable '{var_name}' not found in {mat_path}. "
                           f"Available: {list(f.keys())[:30]} ...")
        data = f[var_name][()]
    # h5py reads MATLAB arrays with the same shape, but note: MATLAB stores column-major.
    # For numeric arrays, the shape matches expectation; we will handle orientation explicitly later.
    return np.array(data)


def ensure_TN(arr: np.ndarray, T: int = 300) -> np.ndarray:
    """
    Ensure array is shaped as (T, N).
    Your MATLAB is (T, N). Sometimes it might come in as (N, T).
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[0] == T:
        return arr
    if arr.shape[1] == T:
        return arr.T
    raise ValueError(f"Cannot interpret array shape {arr.shape} as (T,N) with T={T}.")


def repeat_segment_const(x_TN: np.ndarray) -> np.ndarray:
    """
    Given (T,N) but constant within each segment (in MATLAB you already stored constant),
    return (T,N) where each column is filled with the first value of that segment.
    """
    first = x_TN[0:1, :]           # (1,N)
    return np.repeat(first, x_TN.shape[0], axis=0)  # (T,N)


def main():
    # ====== 配置路径（按你的实际路径改）======
    seg_mat   = r"F:\Matlab_Code_For_DL_data\gdp_drifter_NIW_300h_segments.mat"
    seg_raw_mat   = r"F:\Matlab_Code_For_DL_data\gdp_drifter_raw_uv_300h_segments.mat"
    wind_stress_mat  =  r"F:\Matlab_Code_For_DL_data\drifter_ERA5_u10v10_tx_ty_dtx_dty_300h.mat"

    mld_mat   = r"F:\Matlab_Code_For_DL_data\drifter_oras5_MLD_300h.mat"  # 如果还没有，可暂时注释掉
    slab_mat  = r"F:\Matlab_Code_For_DL_data\drifter_inertial_u_iv_i_fixedpoint_300h_20260112_fororas5.mat"

    out_npz   = r"drifter_niw_300h_new_oras5_mld_raw.npz"
    out_meta  = r"drifter_niw_300h_meta_new_oras5_raw.json"

    T = 300

    # ====== 1) 读真值（漂流浮标带通后的惯性流）======
    uI = ensure_TN(load_mat_var(seg_mat, "uI_300"), T)
    vI = ensure_TN(load_mat_var(seg_mat, "vI_300"), T)

    lon = ensure_TN(load_mat_var(seg_mat, "lon_300"), T)
    lat = ensure_TN(load_mat_var(seg_mat, "lat_300"), T)
    f0  = ensure_TN(load_mat_var(seg_mat, "f_300"), T)

    # 读取原始漂流浮标速度
    u_raw = ensure_TN(load_mat_var(seg_raw_mat, "u_raw_300"), T)
    v_raw = ensure_TN(load_mat_var(seg_raw_mat, "v_raw_300"), T)
     
    # 你要求 lon/lat/f 用一个常数：每段取第一个值
    lon0 = repeat_segment_const(lon)
    lat0 = repeat_segment_const(lat)
    f0c  = repeat_segment_const(f0)

    # ====== 2) 读风速（ERA5取固定点最近格点后逐小时序列）======
    dtx = ensure_TN(load_mat_var(wind_stress_mat, "dtx_300"), T)
    dty = ensure_TN(load_mat_var(wind_stress_mat, "dty_300"), T)
    
    tx = ensure_TN(load_mat_var(wind_stress_mat, "tx_300"), T)
    ty = ensure_TN(load_mat_var(wind_stress_mat, "ty_300"), T)

    # ====== 3) 读MLD（月平均映射到每小时）======
    H = ensure_TN(load_mat_var(mld_mat, "H_300"), T)  # 如果你还没生成就注释这行并改 feature

    # ====== 4) 读slab/板模惯性流（你文件里变量名是 u_i_300 / v_i_300）======
    u_slab = ensure_TN(load_mat_var(slab_mat, "u_i_300"), T)
    v_slab = ensure_TN(load_mat_var(slab_mat, "v_i_300"), T)

    # ====== 5) 统一 Nseg 并裁剪到共同有效样本 ======
    # 以真值为基准
    Nseg = uI.shape[1]
    for name, arr in {
        "vI": vI, "dtx": dtx, "dty": dty, "H": H, "lon0": lon0, "lat0": lat0, "f0": f0c,
        "u_slab": u_slab, "v_slab": v_slab, "tx": tx, "ty": ty, "u_raw": u_raw, "v_raw": v_raw
    }.items():
        if arr.shape[1] != Nseg:
            raise ValueError(f"Nseg mismatch: {name} has {arr.shape[1]} but uI has {Nseg}.")

    # ====== 6) 组装 X, Y ======
    # X: [N, T, F]
    # Y: [N, T, 2]
    # 注意：目前都是 (T,N)，先 stack 后 transpose
    feature_names = ["dtx", "dty", "tx", "ty", "H", "lon", "lat", "f", "u_slab", "v_slab", "u_raw", "v_raw"]
    X_TNF = np.stack([dtx, dty, tx, ty, H, lon0, lat0, f0c, u_slab, v_slab, u_raw, v_raw], axis=-1)  # (T,N,F)
    Y_TN2 = np.stack([uI, vI], axis=-1)                                        # (T,N,2)

    X = np.transpose(X_TNF, (1, 0, 2)).astype(np.float32)  # (N,T,F)
    Y = np.transpose(Y_TN2, (1, 0, 2)).astype(np.float32)  # (N,T,2)

    # ====== 7) 可选：剔除含 NaN 的样本（强烈建议）======
    good = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(Y).all(axis=(1, 2))


    n_bad = int((~good).sum())
    print(f"Total segments: {Nseg}, bad segments (NaN/Inf): {n_bad}")

    X = X[good]
    Y = Y[good]
    print(f"Kept segments: {X.shape[0]}")

    # ====== 8) 保存 npz + meta ======
    np.savez_compressed(out_npz, X=X, Y=Y)

    meta = {
        "T": int(T),
        "feature_names": feature_names,
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "source_files": {
            "segments": str(Path(seg_mat).resolve()),
            "wind": str(Path(wind_stress_mat).resolve()),
            "mld": str(Path(mld_mat).resolve()),
            "slab": str(Path(slab_mat).resolve()),
        },
        "notes": [
            "lon/lat/f are treated as per-segment constants (first value) and repeated to length T.",
            "Nearest-grid sampling already done in MATLAB.",
            "Segments with any NaN/Inf in X or Y are dropped."
        ]
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved NPZ:  {out_npz}")
    print(f"Saved META: {out_meta}")


if __name__ == "__main__":
    main()
