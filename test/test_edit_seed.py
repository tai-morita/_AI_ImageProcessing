# シード点を編集する
# ペイントでシード点を編集するわけだが、複数ピクセルにまたがってしまう。
# 隣接するピクセルは同じシード点になってほしいので、隣接ピクセルは同一ラベルとして統合する。

import re
import glob
import os
import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage import io
from skimage import measure
from skimage.morphology import erosion, ball

def neighbors_lin_2d(coords, arr_shape, neibors=8):
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

def delete_isolated(frames, neibors=6):
    # 隣接６近傍のうち孤立点があれば削除する
    frames_eroded = np.zeros_like(frames, dtype=bool)
    frames_eroded[1:, :, :] |= frames[1:, :, :] == frames[:-1, :, :]
    frames_eroded[:-1, :, :] |= frames[:-1, :, :] == frames[1:, :, :]
    frames_eroded[:, 1:, :] |= frames[:, 1:, :] == frames[:, :-1, :]
    frames_eroded[:, :-1, :] |= frames[:, :-1, :] == frames[:, 1:, :]
    frames_eroded[:, :, 1:] |= frames[:, :, 1:] == frames[:, :, :-1]
    frames_eroded[:, :, :-1] |= frames[:, :, :-1] == frames[:, :, 1:]
    
    frames_isolated = ~frames_eroded

    return frames_isolated

def annotation_to_seed(folder_path, output_path):
    # フォルダ内のアノテーション画像をまとめてseed画像にする
    file_paths = glob.glob(os.path.join(folder_path, 'test*.tif'))
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        match = re.search(r'\d+', basename)
        
        if "test" in basename:
            annotation_count = 500
        else:
            if match is None:
                raise ValueError(f"番号を取得できません: {basename}")
            annotation_count = int(match.group())

            if not (str(annotation_count) in os.path.basename(file_path)):
                print(f"アノテーション番号 {annotation_count} とファイル名が一致していません: {file_path}")
        
        frames = tiff.imread(file_path)
        
        if frames.ndim != 3:
            raise ValueError(f"3Dスタック (Z,Y,X) を想定していますが、shape={frames.shape} です。")
        
        frames_binary = (frames > 0).astype(np.uint16)
        frames_binary_reversed = 65535 - frames_binary * 65535 # 背景とアノテーションを65535にする

        # 背景部分 = 1
        # 歯列 = 0
        # アノテーション部分 > 1 で出力されるので、1 以上の部分は統一させる
        labels = measure.label(frames_binary_reversed, connectivity=2)
        labels[labels == 1] = 0
        labels[labels > 1]  = annotation_count

        # 孤立点を削除する
        labels_isolated = delete_isolated(labels)
        labels_eroded = labels.copy()
        labels_eroded[labels_isolated] = 0

        tiff.imwrite(r"./test\Output/test_labels_eroded.tif", labels_eroded.astype(np.uint16))
        
if __name__ == "__main__":
    annotation_to_seed(r"./test\Input\Markers", None)
    """
    edit_seed_3d(
        input_path="./Watershed/Tooth/mask.tif",
        output_path="./Watershed/Tooth/mask_edited.tif",
        slice_indices=[34, 91, 164]
    )
    add_original_seed(r"./test\Input\mask_edited.tif",
                      r"./test\Input\seed_65.tif",
                      65)
    """