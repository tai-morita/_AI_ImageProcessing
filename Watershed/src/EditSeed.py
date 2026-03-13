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

def edit_seed(volume_label_path, volume_label_edit_path, output_path):
    original_frames = tiff.imread(volume_label_path)
    edit_frames = tiff.imread(volume_label_edit_path)
    seed_frames = original_frames - edit_frames
    seed_index = 0
    for slice_index in range(seed_frames.shape[0]):
        seed_slice = seed_frames[slice_index]
        if np.sum(seed_slice) == 0:
            continue  # シード点がないスライスはスキップ
        label, numbers = ndimage.label(seed_slice)
        print(f"スライス {slice_index} のシード点数: {numbers}")  # デバッグ用
        label[label > 0] += seed_index  # ラベル番号をシード点数分ずらす
        seed_frames[slice_index] = label.astype(np.int16)
        seed_index += numbers
        # tiff.imwrite(f"./study/Watershed/Data/Input/20260313_test/seed_slice_{slice_index}.tif", seed_frames[slice_index])
    print(f"総シード点数: {seed_index}")  # デバッグ用
    tiff.imwrite(output_path, seed_frames.astype(np.int16))
