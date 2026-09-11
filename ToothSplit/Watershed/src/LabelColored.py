import os
import numpy as np
import tifffile as tiff

# ラベル画像をカラー画像に変換する

def _generate_unique_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    indices = np.arange(1, count + 1, dtype=np.uint32)
    color_values = (indices * np.uint32(2654435761)) & np.uint32(0x00FFFFFF)
    colors = np.stack(
        [
            (color_values >> 16) & np.uint32(255),
            (color_values >> 8) & np.uint32(255),
            color_values & np.uint32(255),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3):
        raise ValueError(f"2D/3Dラベル画像を想定していますが、shape={labels.shape} です。")

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    color_image = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if unique_labels.size == 0:
        return color_image

    colors = _generate_unique_colors(unique_labels.size)
    for label_id, color in zip(unique_labels, colors):
        color_image[labels == label_id] = color

    return color_image


def save_colorized_labels(labels: np.ndarray, output_path: str) -> str:
    color_image = colorize_labels(labels)
    tiff.imwrite(output_path, color_image)
    return output_path

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
    for n in [1, 2, 3]:
        input = f"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{n}\splitted_teeth_volume.tif"
        volume = tiff.imread(input)
        output = os.path.join(os.path.dirname(input), "splitted_teeth_volume_colored.tif")
        save_colorized_labels(volume, output)