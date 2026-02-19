# watershed_3d.py
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util

def watershed_3d_tiff(input_path: str, labels_out: str):
    # --- 1) 読み込み（3D）---
    vol = tiff.imread(input_path)  # (Z, Y, X)
    if vol.ndim != 3:
        raise ValueError(f"3Dボリュームを想定していますが、形状が {vol.shape} です。")

    vol = util.img_as_float(vol)

    # --- 2) 前処理 ---
    vol_smooth = filters.gaussian(vol, sigma=1.0, preserve_range=True)

    # --- 3) しきい値で前景抽出 ---
    # 均一ではない照明なら、局所しきい値やOtsuのZ別適用なども検討
    thr = filters.threshold_otsu(vol_smooth)
    bw = vol_smooth > thr
    bw = morphology.remove_small_objects(bw, min_size=64)

    # --- 4) 距離変換（3D） & マーカー ---
    dist = ndi.distance_transform_edt(bw)

    # 3D のピークローカル最大を検出
    # footprint は 3x3x3 の近傍（26近傍）相当
    coords = feature.peak_local_max(
        dist,
        labels=bw,
        footprint=np.ones((3, 3, 3), dtype=bool),
        exclude_border=False
    )

    markers = np.zeros_like(vol, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i

    # （ピークが多すぎる場合）h-maximaを併用
    # markers = morphology.label(morphology.h_maxima(dist, h=1.0))

    # --- 5) 3D watershed ---
    # elevation には負の距離を使うパターンもよく使われます（山＝中心を谷とみなす）
    labels = segmentation.watershed(
        -dist,  # 中心に向かうように
        markers=markers,
        mask=bw,
        connectivity=1  # 3Dの6近傍
    )

    # --- 6) 保存（32bit推奨）---
    print(f"ラベル数: {labels.max()}")  # デバッグ用
    tiff.imwrite(labels_out, labels.astype(np.int16))

if __name__ == "__main__":
    watershed_3d_tiff(
        input_path="./study/test/Input/yn-omusubi_inference_shrink4.tif",
        labels_out="./study/test/Output/yn-omusubi_inference_shrink4_labels_3d.tif"
    )