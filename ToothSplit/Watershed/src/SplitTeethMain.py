# 歯の領域分割
"""
Watershed を用いて歯の領域分割を行う。
Seed は根管を主に使用する。根管が FOV 辺縁に配置されている場合は、隣接歯の接触直前のスライスにする。

処理手順
1. データのインポート: (Original の Volume データ, 歯が前景の 2 値化画像)
2. Volume データでの歯の抽出をする
    Output: overlay 画像
3. Overlay 画像から、疑似 CT 値 のプロファイルから根管の閾値を決定する
4. 根管の閾値を用いて、根管の 2 値化画像を作成する
    Output: 歯が 1, 根管が 2, 背景が 0 の 3 値画像
5. 根管の二値化画像を編集し、 1 Seed 1 Label とする
6. Watershed を用いて歯の領域分割を行う
"""

import os
import numpy as np
import tifffile
import time
from scipy import ndimage as ndi

from DataLoad import load_data, save_volume
from OverlayExtraTooth import overlay_extra_tooth, ternary_image_0B1R2T
from ExtractionRootCanal import extract_root_canal
from LabeledRootCanal import labeled_for_root_canal
from EditRootCanalLabel import edit_root_canal
from WaterShedBySkimage import watershed_3d_tiff
from PointPlotFotGPU import load_npy_volume, volume_to_point_cloud, show_gpu_point_cloud
from Graph import create_seed_main

def SplitTeethMain(volume_file_path: str, teeth_binary_file_path: str):
    """
    Watershed を用いて歯の領域分割を行うメイン関数
    input: 
        - volume_file_path      : 元の Volume データ(.npy)のファイルパス
        - teeth_binary_file_path: 歯が前景の 2 値化画像(.npy)のファイルパス
    output:
        - extracted_volume         : Volume データでの歯の抽出結果
        - edited_labeled_root_canal: 編集済みのラベル付き根管画像
        - splitted_teeth_volume    : Watershed による歯の領域分割
    """
    # Step1: データのインポート
    volume       = load_data(volume_file_path)
    teeth_binary = load_data(teeth_binary_file_path)

    # Step2: Volume データでの歯の抽出をする
    extracted_volume = overlay_extra_tooth(volume, teeth_binary)

    # Step3: Overlay 画像から、疑似 CT 値 のプロファイルから根管の閾値を決定し、二値化で根幹を抽出する
    root_canal_binary, threshold = extract_root_canal(extracted_volume)
    # ternary_image = ternary_image_0B1R2T(extracted_volume, root_canal_binary)

    # Step4: 根管を基準とした Seed を作成する
    # 元の Volume と根管に対して Connected Component Labeling を行う。
    # これらのラベル付き Volume をもとにグラフを作成し、 Seed を決定する。
    labeled_root_canal, _     = ndi.label(root_canal_binary)
    edited_labeled_root_canal = edit_root_canal(labeled_root_canal, volume)
    seed_root_canal           = create_seed_main(teeth_binary, edited_labeled_root_canal, debug=False)

    # Step5: Watershed を用いて歯の領域分割を行う
    splitted_teeth_volume = watershed_3d_tiff(
        extracted_volume,
        seed_root_canal,
        output_path=None,
        connectivity=26,
        debug_dist_path=None)

    return extracted_volume, edited_labeled_root_canal, splitted_teeth_volume


def main():
    label_number = 3
    input_volume_path       = fr"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{label_number}\CTHRs_100_Label{label_number}.npy"
    input_teeth_binary_path = fr"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{label_number}\Label{label_number}.npy"
    extracted_volume, edited_labeled_root_canal, splitted_teeth_volume = SplitTeethMain(input_volume_path, input_teeth_binary_path)
    # 結果の保存
    output_dir = r"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\temp"
    if False:
        save_volume(os.path.join(output_dir, "extracted_volume.npy")         , extracted_volume)
        save_volume(os.path.join(output_dir, "edited_labeled_root_canal.npy"), edited_labeled_root_canal, dtype = np.int8)
        save_volume(os.path.join(output_dir, "splitted_teeth_volume.npy")    , splitted_teeth_volume    , dtype = np.int8)
    if True:
        save_volume(os.path.join(output_dir, os.path.basename(input_volume_path).replace(".npy", ".tif"))      , load_data(os.path.join(output_dir, input_volume_path))      , dtype = np.float32)
        save_volume(os.path.join(output_dir, os.path.basename(input_teeth_binary_path).replace(".npy", ".tif")), load_data(os.path.join(output_dir, input_teeth_binary_path)), dtype = np.uint16)
        save_volume(os.path.join(output_dir, "extracted_volume.tif")         , extracted_volume, dtype = np.uint16)
        save_volume(os.path.join(output_dir, "edited_labeled_root_canal.tif"), edited_labeled_root_canal, dtype = np.uint16)
        save_volume(os.path.join(output_dir, "splitted_teeth_volume.tif")    , splitted_teeth_volume, dtype = np.uint16)



if __name__ == "__main__":
    main()