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
    date = "20260611"
    No = 8
    keyword = "_filled"
    connectivity = 26
    InputDir   = r".\Watershed\Data\Input\labeled_map"
    OutputDir  = r".\Watershed\Data\Output"
    input_path            = r".\Watershed\Data\Input\labeled_map_filled\label_map_8_filled.tif"
    volume_label_original = r".\Watershed\Data\Output\label_map_8_filled.tif"
    volume_label_annotate = r".\Watershed\Data\Output\label_map_8_filled_annotate.tif"
    seed_path             = r".\Watershed\Data\Output\volume_8_filled_seed.tif"
    output_path           = r".\Watershed\Data\Output\watershed_volume_8_filled_conn=26.tif"
    # edit_seed(volume_label_original, volume_label_annotate, seed_path)
    # edit_seed_1teeth_per_slice(volume_label_original, volume_label_annotate, seed_path)
    test_edit_seed_main(input_path, volume_label_annotate)  # これでアノテーションGUIが起動するので、そこで編集して保存する
    output_path = r".\Watershed\Data\Output\test\test.tif"
    watershed_3d_tiff(input_path, volume_label_annotate, output_path, connectivity=connectivity)
