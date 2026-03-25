import glob
import os

import tifffile as tiff
from scipy import ndimage as ndi
from skimage import morphology


def fill_holes_in_binary_volume(
    input_path: str,
    output_path: str,
    distance_output_path: str | None = None,
    per_slice: bool = False,
    min_hole_size: int | None = None,
) -> None:
    volume = tiff.imread(input_path)
    bw = volume > 0

    if per_slice:
        filled = bw.copy()
        for slice_index in range(bw.shape[0]):
            filled[slice_index] = ndi.binary_fill_holes(bw[slice_index])
    else:
        filled = ndi.binary_fill_holes(bw)

    if min_hole_size is not None:
        filled = morphology.remove_small_holes(filled, area_threshold=min_hole_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, filled.astype("uint8"))

    if distance_output_path is not None:
        distance = ndi.distance_transform_edt(filled)
        os.makedirs(os.path.dirname(distance_output_path), exist_ok=True)
        tiff.imwrite(distance_output_path, distance.astype("float32"))

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    if distance_output_path is not None:
        print(f"Distance Output: {distance_output_path}")
    print(f"Filled voxels: {(filled.sum() - bw.sum())}")


if __name__ == "__main__":
    input_dir = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\labeled_map"
    output_dir = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\20260323"

    files = [
        file_path
        for file_path in glob.glob(os.path.join(input_dir, "label_map_3.tif"))
        if "annotate" not in os.path.basename(file_path).lower()
        and "dist" not in os.path.basename(file_path).lower()
    ]

    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(input_dir, f"{base_name}_filled.tif")
        # distance_output_path = os.path.join(input_dir, f"{base_name}_filled_dist.tif")
        print(f"Processing: {file_path}")
        fill_holes_in_binary_volume(
            input_path=file_path,
            output_path=output_path,
            distance_output_path=None,
            per_slice=False,
            min_hole_size=None,
        )