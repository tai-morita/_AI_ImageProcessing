# npy or tiff ファイルを numpy 形式で読み込む
import tifffile
import numpy as np
import os

def load_data(input_file_path: str):
    file_name, file_ext = os.path.splitext(input_file_path)
    if file_ext == ".tif":
        volume = tifffile.imread(input_file_path)
    elif file_ext == ".npy":
        volume = np.load(input_file_path)
    else:
        raise ValueError(f"対応していないファイル形式です: file_name: {file_name}, file_ext: {file_ext}")
    return volume

def save_volume(output_file_path: str, volume: np.ndarray, dtype=None):
    if dtype is None:
        dtype = volume.dtype
    print(f"dtype: {dtype}, save file: {output_file_path}")
    file_name, file_ext = os.path.splitext(output_file_path)
    if file_ext == ".tif":
        volume = tifffile.imwrite(output_file_path, volume.astype(dtype))
    elif file_ext == ".npy":
        volume = np.save(output_file_path, volume.astype(dtype))
    else:
        raise ValueError(f"対応していないファイル形式です: file_name: {file_name}, file_ext: {file_ext}")

if __name__ == "__main__":
    path = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\label_1\CTHRs_100_Label1.npy"
    volume = load_data(path)

    save_volume(r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\label_1\CTHRs_100_Label1.tif", volume)