import tifffile as tiff
import numpy as np
from scipy import ndimage
from skimage import io
try:
    from .Watershed_skimage import watershed_3d_tiff
    from .EditSeed import edit_seed, edit_seed_1teeth_per_slice
    from .EditSeed_test import test_edit_seed_main
    from .AnnotationONGUI import main as annotation_main
except ImportError:
    from Watershed_skimage import watershed_3d_tiff
    from EditSeed import edit_seed, edit_seed_1teeth_per_slice
    from EditSeed_test import test_edit_seed_main
    from AnnotationONGUI import main as annotation_main

"""
Usage
InputDir
OutputDir
input_path: 歯と背景の二値化画像
volume_label_original: input_path と同じ (seed の編集のために必要)
volume_label_annotate: 編集後のラベルマップ
seed_path: シード画像: watershed の入力として使用する
output_path: 出力先のパス

watershed で必要なもの
- input_path: 二値化画像
- seed_path: シード画像
"""


if __name__ == "__main__":

    date = "20260421"
    No = 1
    keyword = "_filled"
    connectivity = 26
    InputDir   = f"./Watershed/Data/Input/{date}{keyword}255_No{No}"
    OutputDir  = f"./Watershed/Data/Output/{date}"
    input_path            = f"{InputDir}/label_map_{No}{keyword}.tif"
    volume_label_original = f"{InputDir}/label_map_{No}{keyword}.tif"
    volume_label_annotate = f"{InputDir}/label_map_{No}{keyword}_annotate.tif"
    seed_path             = f"{InputDir}/volume_{No}{keyword}_seed.tif"
    output_path           = f"{OutputDir}/watershed_volume_{No}{keyword}_conn={connectivity}.tif"
    # edit_seed(volume_label_original, volume_label_annotate, seed_path)
    # edit_seed_1teeth_per_slice(volume_label_original, volume_label_annotate, seed_path)
    test_edit_seed_main(volume_label_original, volume_label_annotate)  # これでアノテーションGUIが起動するので、そこで編集して保存する
    watershed_3d_tiff(input_path, volume_label_annotate, output_path, connectivity=connectivity)
