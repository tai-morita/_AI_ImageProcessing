# watershed_3d.py
import os
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util
from skimage.color import label2rgb

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
    # --- 1) 読み込み（3D）---
    vol = tiff.imread(input_path)  # (Z, Y, X)
    if vol.ndim != 3:
        raise ValueError(f"3Dボリュームを想定していますが、形状が {vol.shape} です。")

    vol = util.img_as_float(vol)

    # --- 2) 前処理 ---
    vol_smooth = filters.gaussian(vol, sigma=1.0, preserve_range=True)

    # --- 3) しきい値で前景抽出 ---
    # 均一ではない照明なら、局所しきい値やOtsuのZ別適用なども検討
    # 今回は前景をすでに抽出したものを使うので、いらない
    bw = vol_smooth > 0
    """
    thr = filters.threshold_otsu(vol_smooth)
    bw = vol_smooth > thr
    bw = morphology.remove_small_objects(bw, min_size=64)
    """

    # --- 4) 距離変換（3D） & マーカー ---
    dist = ndi.distance_transform_edt(bw)
    # tiff.imwrite("./test/Output/dist.tif", dist.astype(np.float32))  # デバッグ用

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

    # --- 6) 保存（32bit推奨）---
    print(f"ラベル数: {labels.max()}")  # デバッグ用
    tiff.imwrite(labels_out, labels.astype(np.int16))

def merge_volume(axial_path, sagittal_path, coronal_path):
    # 3つのデータを統合させる
    # axial面にして比較する
    frames_axial    = tiff.imread(axial_path)
    frames_sagittal = tiff.imread(sagittal_path)
    frames_coronal  = tiff.imread(coronal_path)

    # frames_sagittal = np.transpose(frames, (2, 0, 1))
    frames_sagittal_to_axial = np.transpose(frames_sagittal, (1, 2, 0))
    # frames_coronal = np.transpose(frames, (1, 0, 2))
    frames_coronal_to_axial = np.transpose(frames_coronal, (1, 0, 2))

    # 

    dir = os.path.dirname(axial_path)
    # tiff.imwrite(os.path.join(dir, "temp_sagittal.tif"), frames_sagittal_to_axial.astype(np.uint16))
    # tiff.imwrite(os.path.join(dir, "temp_coronal.tif"), frames_coronal_to_axial.astype(np.uint16))



if __name__ == "__main__":
    """
    watershed_3d_tiff(
        input_path  = "./test/Input/data_053.tif",
        markers_path= "./test/Input/mask_edited.tif",
        labels_out  = "./test/Output/Watershed_data_053.tif"
    )
    """
    merge_volume(r"./test\Output\Watershed_data_053.tif",
                 r"./test\Output\Watershed_data_053_sagittal.tif",
                 r"./test\Output\Watershed_data_053_coronal.tif")