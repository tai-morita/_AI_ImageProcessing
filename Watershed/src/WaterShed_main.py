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
    volume_label_original = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\Tooth_label.tif"
    volume_label_annotate = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\testWS\Tooth_label_annotate.tif"
    seed_path             = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\testWS\Tooth_label_seed.tif"
    input_path            = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\Tooth_label.tif"
    output_path           = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\testWS\watershed_Tooth_label_conn=26.tif"
    connectivity = 26
    test_edit_seed_main(input_path, seed_path)  # これでアノテーションGUIが起動するので、そこで編集して保存する
    # edit_seed_1teeth_per_slice(volume_label_original, volume_label_annotate, seed_path)
    # edit_seed(volume_label_original, volume_label_annotate, seed_path)
    watershed_3d_tiff(input_path, seed_path, output_path, connectivity=connectivity)
