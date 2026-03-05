import os
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, util
from skimage.color import label2rgb

def merge_volume(axial_path, sagittal_path, coronal_path):
    # 3つのデータを統合させる
    # axial面にして比較する
    frames_axial    = tiff.imread(axial_path)
    frames_sagittal = tiff.imread(sagittal_path)
    frames_coronal  = tiff.imread(coronal_path)

    # frames_sagittal = np.transpose(frames, (2, 0, 1))
    # frames_coronal = np.transpose(frames, (1, 0, 2))
    frames_sagittal_to_axial = np.transpose(frames_sagittal, (1, 2, 0))
    frames_coronal_to_axial = np.transpose(frames_coronal, (1, 0, 2))

    dir = os.path.dirname(axial_path)
    tiff.imwrite(os.path.join(dir, "temp_sagittal.tif"), frames_sagittal_to_axial.astype(np.uint16))
    tiff.imwrite(os.path.join(dir, "temp_coronal.tif"), frames_coronal_to_axial.astype(np.uint16))



if __name__ == "__main__":
    """
    watershed_3d_tiff(
        input_path  = "./Data/Input/data_053.tif",
        markers_path= "./Data/Input/mask_edited.tif",
        labels_out  = "./Data/Output/Watershed_data_053.tif"
    )
    """
    merge_volume(r"./Data/Output/Watershed_data_053.tif",
                 r"./Data/Output/Watershed_data_053_sagittal.tif",
                 r"./Data/Output/Watershed_data_053_coronal.tif")