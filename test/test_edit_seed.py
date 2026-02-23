# シード点を編集する
# ペイントでシード点を編集するわけだが、複数ピクセルにまたがってしまう。
# 隣接するピクセルは同じシード点になってほしいので、隣接ピクセルは同一ラベルとして統合する。

import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage import io

def neighbors_lin(coords, arr_shape, neibors=8):
    # 8近傍の座標を返す
    # ただし、画像の範囲外の座標は返さない (1ライン増やすのとどっちがのであろう)
    height = coords[0]
    width = coords[1]
    if neibors == 4:
        coords_neibor = [(height-1,  width), (height+1, width), 
                (height, width-1), (height, width+1)]
    elif neibors == 8:
        coords_neibor = [(height-1, width-1), (height-1, width), 
                (height-1, width+1), (height, width-1), 
                (height, width+1), (height+1, width-1), 
                (height+1, width), (height+1, width+1)]
    # 画像の範囲外の座標は返さない
    coords_neibor = [coord for coord in coords_neibor if 0 <= coord[0] < arr_shape[0] and 0 <= coord[1] < arr_shape[1]]
    return coords_neibor

def edit_seed_3d(input_path: str, output_path: str, slice_indices: list):
    # --- 1) 読み込み（3d）---
    # インプットは根尖を抽出したボリュームデータ
    # ラベリングにするスライスを複数指定して、それらをラベリングするようにする。
    if input_path.endswith('.tif'):
        vol = tiff.imread(input_path)  # (Y, X)
    else:
        vol = io.imread(input_path)  # (Y, X)
    if vol.ndim != 3:
        raise ValueError(f"2D画像を想定していますが、形状が {vol.shape} です。")

    label_number = 0
    arr3d_label = np.zeros_like(vol, dtype=np.int32)
    for slice_index in slice_indices:
        # スライスごとにラベリングする
        arr2d_slice = vol[slice_index]
        arr2d_label, arr2d_numbers = ndimage.label(arr2d_slice)
        print(f"スライス {slice_index} のラベル数: {arr2d_numbers}")  # デバッグ用
        arr2d_label[arr2d_label > 0] += label_number
        arr3d_label[slice_index] = arr2d_label
        label_number += arr2d_numbers
    print(f"ラベル数: {label_number}")  # デバッグ用
    tiff.imwrite(output_path, arr3d_label.astype(np.int16))

def add_original_seed(seed_path, add_seed_path, frame_number):
    # 手動で追加したseed画像を挿入する
    seed_frames = tiff.imread(seed_path)
    add_seed_frame = tiff.imread(add_seed_path)
    # seedが連番になるように調節
    max_count = np.max(seed_frames)
    add_seed_frame[add_seed_frame > 0] += max_count
    seed_frames[frame_number] = add_seed_frame
    tiff.imwrite(seed_path, seed_frames.astype(np.uint16))

if __name__ == "__main__":
    """
    edit_seed_3d(
        input_path="./Watershed/Tooth/mask.tif",
        output_path="./Watershed/Tooth/mask_edited.tif",
        slice_indices=[34, 91, 164]
    )
    """
    add_original_seed(r"./test\Input\mask_edited.tif",
                      r"./test\Input\seed_65.tif",
                      65)