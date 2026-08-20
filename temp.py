import numpy as np
from scipy.signal import savgol_filter

def detect_flat_region(x, y, smooth_window=11, poly=2,
                       slope_k=1.5, curve_k=1.5, min_len=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1) smooth
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = min(smooth_window, len(y) - (1 - len(y) % 2))
    ys = savgol_filter(y, smooth_window, poly)

    # 2) derivatives
    dy = np.gradient(ys, x)
    d2y = np.gradient(dy, x)

    # ノイズスケールをMADで推定
    mad_dy = np.median(np.abs(dy - np.median(dy))) + 1e-12
    mad_d2 = np.median(np.abs(d2y - np.median(d2y))) + 1e-12

    flat_mask = (np.abs(dy) < slope_k * mad_dy) & (np.abs(d2y) < curve_k * mad_d2)

    # 3) 連続区間抽出
    idx = np.where(flat_mask)[0]
    if len(idx) == 0:
        return None

    segments = []
    s = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            e = idx[i-1]
            if e - s + 1 >= min_len:
                segments.append((s, e))
            s = idx[i]
    e = idx[-1]
    if e - s + 1 >= min_len:
        segments.append((s, e))

    if not segments:
        return None

    # 4) 最初の主山の手前の最長区間を採用
    peak_idx = int(np.argmax(ys))
    pre_peak_segments = [seg for seg in segments if seg[1] < peak_idx]
    target = max(pre_peak_segments if pre_peak_segments else segments, key=lambda t: t[1]-t[0])

    i0, i1 = target
    return {
        "index_range": (i0, i1),
        "x_range": (x[i0], x[i1]),
        "y_mean": float(np.mean(y[i0:i1+1]))
    }

if __name__ == "__main__":
    import tifffile

    volume = tifffile.imread(r"./study/Watershed/t-oe/20260716_NR_Label/label_2/CTHRs_100_Label2_overlay.tif")

    # 0 is background, so only non-zero voxels are used for histogram analysis.
    values = volume[volume != 0].astype(np.float64)

    if values.size == 0:
        print("Non-zero voxel was not found.")
        raise SystemExit(1)

    bins = 256
    y, edges = np.histogram(values, bins=bins)
    x = 0.5 * (edges[:-1] + edges[1:])

    result = detect_flat_region(
        x,
        y,
        smooth_window=11,
        poly=2,
        slope_k=1.5,
        curve_k=1.5,
        min_len=8,
    )

    print(f"volume shape: {volume.shape}")
    print(f"non-zero voxels: {values.size}")
    print(f"hist bins: {bins}")

    if result is None:
        print("Flat region was not detected.")
    else:
        i0, i1 = result["index_range"]
        x0, x1 = result["x_range"]
        print("Flat region detected")
        print(f"  index range: ({i0}, {i1})")
        print(f"  value range: ({x0:.6g}, {x1:.6g})")
        print(f"  mean histogram count: {result['y_mean']:.6g}")
