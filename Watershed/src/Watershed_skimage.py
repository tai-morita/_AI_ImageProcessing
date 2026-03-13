# watershed_3d.py
import os
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util
try:
    from .RotateVolume import axial_transpose
    from .FileOperation import save_colorized_labels
except ImportError:
    from RotateVolume import axial_transpose
    from FileOperation import save_colorized_labels

def threshold_input_tiff(input_path: str, output_path: str):
    # --- 1) 読み込み（3D）---
    vol = tiff.imread(input_path)  # (Z, Y, X)
    if vol.ndim != 3:
        raise ValueError(f"3Dボリュームを想定していますが、形状が {vol.shape} です。")

    # --- 2) 前処理 ---
    vol_smooth = filters.gaussian(vol, sigma=1.0, preserve_range=True)

    # --- 3) しきい値で前景抽出 ---
    #背景はすでに0なので、前景を1にするだけ
    bw = vol_smooth > 0

    tiff.imwrite(output_path, bw.astype(np.uint8) * 255)

def watershed_3d_tiff(input_path: str, markers_path: str, labels_out: str):
    # インプット: 歯のみの画像, マーカー
    # --- 1) 読み込み（3D）---
    vol = tiff.imread(input_path)  # (Z, Y, X)
    if vol.ndim != 3:
        raise ValueError(f"3Dボリュームを想定していますが、形状が {vol.shape} です。")
    print(f"入力ボリュームの形状: {vol.shape}, データ型: {vol.dtype}")  # デバッグ用
    vol = util.img_as_float(vol)

    # --- 2) 前処理 ---
    vol_smooth = filters.gaussian(vol, sigma=1.0, preserve_range=True)

    # --- 3) しきい値で前景抽出 ---
    # 均一ではない照明なら、局所しきい値やOtsuのZ別適用なども検討
    # 今回は前景をすでに抽出したものを使うので、いらない
    bw = vol > 0
    # tiff.imwrite("./test/Output/binary_mask.tif", bw.astype(np.uint8) * 255)  # デバッグ用
    """
    thr = filters.threshold_otsu(vol_smooth)
    bw = vol_smooth > thr
    bw = morphology.remove_small_objects(bw, min_size=64)
    """

    # --- 4) 距離変換（3D） & マーカー ---
    dist = ndi.distance_transform_edt(bw)
    # tiff.imwrite("./study/Watershed/Data/Output/20260313/dist.tif", dist.astype(np.float32))  # デバッグ用
    # tiff.imwrite("./study/Watershed/Data/Output/20260313/bw.tif", bw.astype(np.float32))  # デバッグ用

    # 3D のピークローカル最大を検出
    # footprint は 3x3x3 の近傍（26近傍）相当
    coords = feature.peak_local_max(
        dist,
        labels=bw,
        footprint=np.ones((3, 3, 3), dtype=bool),
        exclude_border=False
    )

    markers = tiff.imread(markers_path)  # (Z, Y, X)

    # （ピークが多すぎる場合）h-maximaを併用
    # markers = morphology.label(morphology.h_maxima(dist, h=1.0))

    # --- 5) 3D watershed ---
    # elevation には負の距離を使うパターンもよく使われます（山＝中心を谷とみなす）
    labels = segmentation.watershed(
        -dist,  # 中心に向かうように
        markers=markers,
        mask=bw,
        connectivity=6,  # 3Dの6近傍
        watershed_line=False
    )

    # --- 6) 保存（uint16グレースケール & カラー）---
    print(f"ラベル数: {labels.max()}")  # デバッグ用
    # グレースケール保存（uint16）
    tiff.imwrite(labels_out, labels.astype(np.uint16))
    print(f"Saved: {labels_out}, 形状: {labels.shape}, データ型: {labels.dtype}")  # デバッグ用
    base, ext = os.path.splitext(labels_out)
    color_out = base + '_color' + ext
    save_colorized_labels(labels, color_out)
    # axial_transpose(labels_out, os.path.dirname(labels_out))  # Sagittal, Coronal 面も保存
    # axial_transpose(color_out, os.path.dirname(color_out))  # Sagittal, Coronal 面も保存

if __name__ == "__main__":
    watershed_3d_tiff(
        input_path  = "./study/Watershed/Data/Input/20260313_test/volume_1.tif",
        markers_path= "./study/Watershed/Data/Input/20260313_test/label_map_1_annotate.tif",
        labels_out  = "./study/Watershed/Data/Output/20260313/watershed_volume_1.tif"
    )
