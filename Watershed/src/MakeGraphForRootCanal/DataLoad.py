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

def save_volume(output_file_path: str, volume: np):
    print(f"save file: {output_file_path}")
    file_name, file_ext = os.path.splitext(output_file_path)
    if file_ext == ".tif":
            volume = tifffile.imwrite(output_file_path, volume.astype(np.uint16))
    elif file_ext == ".npy":
        volume = np.save(output_file_path, volume.astype(np.uint16))
    else:
        raise ValueError(f"対応していないファイル形式です: file_name: {file_name}, file_ext: {file_ext}")