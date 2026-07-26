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
    InputDir   = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\labeled_map"
    OutputDir  = r"D:\_study\ImageProcessing\study\Watershed\Data\Output"
    input_path            = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\radial_volume.tif"
    volume_label_annotate = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\radial_volume_annotate.tif"
    seed_path             = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\radial_volume_seed.tif"
    output_path           = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\watershed_radial_volume_conn=26.tif"
    # edit_seed(volume_label_original, volume_label_annotate, seed_path)
    # edit_seed_1teeth_per_slice(volume_label_original, volume_label_annotate, seed_path)
    # test_edit_seed_main(input_path, volume_label_annotate)  # これでアノテーションGUIが起動するので、そこで編集して保存する
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\test.tif"
    volume_label_original = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume.tif"
    volume_label_annotate = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume_annotate.tif"
    input_path = volume_label_original
    seed_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume_seed.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\watershed_edited_volume_conn=26.tif"
    edit_seed(volume_label_original, volume_label_annotate, seed_path)
    watershed_3d_tiff(input_path, seed_path, output_path, connectivity=connectivity)
