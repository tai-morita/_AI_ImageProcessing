# watershed_3d.py
import os
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util
from ..FileOperation import save_colorized_labels
from .DataLoad import load_data, save_volume

def watershed_3d_tiff(volume          : np.ndarray, 
                      seed_volume     : np.ndarray, 
                      output_path     : str,
                      connectivity    : int = 6,
                      debug_dist_path : str = None) -> None:
    # インプット: 歯のみの画像, マーカー
    volume = util.img_as_float(volume)

    # --- 1) 距離変換(3D)
    dist = ndi.distance_transform_edt(volume)
    # dist = dist_seed(volume, seed_volume)
    if debug_dist_path:
        save_volume(debug_dist_path, dist)

    # 3D のピークローカル最大を検出
    # footprint は 3x3x3 の近傍（26近傍）相当
    if seed_volume is None:
        coords = feature.peak_local_max(
            dist,
            labels=volume > 0,
            footprint=np.ones((3, 3, 3), dtype=bool),
            exclude_border=False
        )

    # --- 5) 3D watershed ---
    # elevation には負の距離を使うパターンもよく使われます（山＝中心を谷とみなす）
    labels = segmentation.watershed(
        -dist,  # 中心が谷になるように負の距離を使う
        markers=seed_volume,
        mask=volume > 0,
        connectivity=connectivity,  # 3Dの6近傍
        watershed_line=False
    )

    # --- 6) 保存（uint16グレースケール & カラー）---
    print(f"ラベル数: {labels.max()}")  # デバッグ用
    # グレースケール保存（uint16）
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_volume(output_path, labels.astype(np.uint16))
    print(f"Saved: {output_path}")
    print(f"形状: {labels.shape}, データ型: {labels.dtype}")  # デバッグ用
    for i in range(1, labels.max() + 1):
        print(f"ラベル {i}: ボクセル数 = {np.sum(labels == i)}")  # デバッグ用
    base, ext = os.path.splitext(output_path)
    color_out = base + '_color' + ext
    save_colorized_labels(labels, color_out)

    return print("Done.")


def dist_seed(volume: np.ndarray, seed_volume: np.ndarray) -> np.ndarray:
    """前景領域の各ボクセルから最も近いseedまでの距離を返す。

    ``volume > 0`` を距離の対象領域、``seed_volume > 0`` をseedとして
    扱う。対象領域外の距離は0とする。
    """
    if volume.shape != seed_volume.shape:
        raise ValueError(
            "volume and seed_volume must have the same shape: "
            f"{volume.shape} != {seed_volume.shape}"
        )

    foreground_mask = volume > 0
    seed_mask = (seed_volume > 0) & foreground_mask
    if not np.any(seed_mask):
        raise ValueError("seed_volume must contain at least one seed inside volume.")

    distance = ndi.distance_transform_edt(~seed_mask).astype(np.float32)
    distance[~foreground_mask] = 0.0
    return distance