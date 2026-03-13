import tifffile as tiff
import numpy as np
from scipy import ndimage
from skimage import io
try:
    from .Watershed_skimage import watershed_3d_tiff
    from .EditSeed import edit_seed
except ImportError:
    from Watershed_skimage import watershed_3d_tiff
    from EditSeed import edit_seed




if __name__ == "__main__":
    date = "20260313"
    InputDir   = f"./study/Watershed/Data/Input/{date}_test"
    OutputDir  = f"./study/Watershed/Data/Output/{date}"
    input_path            = f"{InputDir}/label_map_1.tif"
    volume_label_original = f"{InputDir}/label_map_1.tif"
    volume_label_edited   = f"{InputDir}/label_map_1_annotate.tif"
    seed_path             = f"{InputDir}/volume_1_seed.tif"
    output_path           = f"{OutputDir}/watershed_volume_1.tif"
    edit_seed(volume_label_original, volume_label_edited, seed_path)
    watershed_3d_tiff(input_path, seed_path, output_path)
