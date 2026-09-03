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

from .DataLoad import load_data, save_volume
from .OverlayExtraTooth import overlay_extra_tooth, ternary_image_0B1R2T
from .ExtractionRootCanal import extract_root_canal
from .LabeledRootCanal import labeled_for_root_canal
from .test_SeedRootcanal import edit_root_canal
from .WaterShedBySkimage import watershed_3d_tiff
from .PointPlotFotGPU import load_npy_volume, volume_to_point_cloud, show_gpu_point_cloud
from .Graph_rev2 import create_seed_main



def main(label_number: int = 1,
         input_dir: str = None, 
         output_dir: str = None,
         debug: bool = False,
         save_middle_data: bool = True,
         save_3d_plot: bool = True) -> None:
    
    # debug = True
    # save_middle_data = False
    # save_3d_plot = True

    if input_dir is None:
        input_dir = rf"./study/Watershed/t-oe/20260716_NR_Label/test/Label{label_number}"
    if label_number is None:
        label_number = 1
    if output_dir is None:
        output_dir = input_dir

    volume_file       = rf"CTHRs_100_Label{label_number}.npy"
    teeth_binary_file = rf"Label{label_number}.npy"


    # Step1: データのインポート
    volume       = load_data(os.path.join(input_dir, volume_file))
    teeth_binary = load_data(os.path.join(input_dir, teeth_binary_file))
    if save_middle_data:
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + ".tif"), volume)
        save_volume(os.path.join(output_dir, os.path.splitext(teeth_binary_file)[0] + ".tif"), teeth_binary)

    # Step2: Volume データでの歯の抽出をする
    extracted_volume = overlay_extra_tooth(volume, teeth_binary)
    if True:
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_overlay.tif"), extracted_volume, dtype=np.uint16)
    # Step3: Overlay 画像から、疑似 CT 値 のプロファイルから根管の閾値を決定し、二値化で根幹を抽出する
    root_canal_binary, threshold = extract_root_canal(extracted_volume)
    if debug:
        print(f"根管の閾値: {threshold}")
    if True:
        # 根管, 歯, 背景の 3 値画像を保存する
        ternary_image = ternary_image_0B1R2T(extracted_volume, root_canal_binary)
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_ternary.tif"), ternary_image, dtype=np.uint8)

    # Step4: 根管を基準とした Seed を作成する
    # 元の Volume と根管に対して Connected Component Labeling を行う。
    # これらのラベル付き Volume をもとにグラフを作成し、 Seed を決定する。
    # labeled_root_canal = labeled_for_root_canal(root_canal_binary, dilation_iterations=1)
    labeled_root_canal, _ = ndi.label(root_canal_binary)
    edited_labeled_root_canal = edit_root_canal(labeled_root_canal, volume)

    if save_middle_data:
        print(f"ラベル数: {np.max(labeled_root_canal)} -> {np.max(edited_labeled_root_canal)}")
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_labeled_root_canal.tif"), labeled_root_canal, dtype=np.uint16)
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_labeled_root_canal_edited.tif"), edited_labeled_root_canal, dtype=np.uint16)

    seed_root_canal = create_seed_main(teeth_binary, edited_labeled_root_canal, debug=debug)
    if save_middle_data:
        print(f"ラベル数: {np.max(seed_root_canal)}")
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_seed.tif"), seed_root_canal, dtype=np.uint16)

    # Step5: Watershed を用いて歯の領域分割を行う
    if save_middle_data and debug:
        debug_dist_path = os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_distance.tif")
    else:
        debug_dist_path = None
    splitted_teeth_volume = watershed_3d_tiff(
        extracted_volume,
        seed_root_canal,
        os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_watershed.tif"),
        debug_dist_path=debug_dist_path
    )
    # Step6: GPU で 3D プロットする
    if save_3d_plot:
        xyz, values = volume_to_point_cloud(splitted_teeth_volume, threshold=0.0, max_points=400000)
        show_gpu_point_cloud(xyz, values, 
                            point_size=1.5, 
                            opacity=0.6, 
                            output_html=os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_3dplot.html"))

    # numpy データを保存する
    if save_numpy:
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_overlay.npy"), extracted_volume, dtype=np.float32)
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_edited_labeled_root_canal.npy"), edited_labeled_root_canal, dtype=np.uint16)
        save_volume(os.path.join(output_dir, os.path.splitext(volume_file)[0] + "_splitted_teeth_volume.npy"), splitted_teeth_volume, dtype=np.uint16)

if __name__ == "__main__":
    debug            = False
    save_middle_data = False
    save_3d_plot     = True
    save_numpy       = True
    print(f"debug: {debug}, save_middle_data: {save_middle_data}, save_3d_plot: {save_3d_plot}")
    for label_number in range(1, 4):
        if label_number != 2:
            # continue
            pass
        print(f"Processing Label {label_number}...")
        input_dir  = fr"./Watershed/data/Label_{label_number}"
        output_dir = None
        start_time = time.time()
        main(label_number=label_number, input_dir=input_dir, output_dir=output_dir, debug=debug, save_middle_data=save_middle_data, save_3d_plot=save_3d_plot)
        end_time = time.time()
        print(f"処理時間: {end_time - start_time:.2f} 秒")
        print("Done.")