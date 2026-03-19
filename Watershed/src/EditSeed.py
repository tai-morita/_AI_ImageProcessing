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


def _build_seed_centroid_volume(seed_labels: np.ndarray) -> np.ndarray:
    centroid_volume = np.zeros_like(seed_labels, dtype=np.int16)
    structure = ndimage.generate_binary_structure(3, 1)

    for label_id in np.unique(seed_labels):
        if label_id == 0:
            continue

        coords = np.argwhere(seed_labels == label_id)
        if coords.size == 0:
            continue

        centroid = coords.mean(axis=0)
        nearest_index = np.argmin(np.sum((coords - centroid) ** 2, axis=1))
        centroid_coord = tuple(coords[nearest_index])

        point_volume = np.zeros_like(seed_labels, dtype=bool)
        point_volume[centroid_coord] = True
        point_volume = ndimage.binary_dilation(point_volume, structure=structure, iterations=1)
        centroid_volume[point_volume] = label_id

        print(
            f"ラベル {label_id}: 重心(連続値) = ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
            f"保存座標 = {centroid_coord}"
        )

    return centroid_volume


def edit_seed(volume_label_path, volume_label_edit_path, output_path):
    original_frames = tiff.imread(volume_label_path)
    edit_frames = tiff.imread(volume_label_edit_path)
    seed_frames = original_frames - edit_frames

    seed_index = 0  # ラベル番号の開始
    for slice_index in range(1, seed_frames.shape[0]+1):
        seed_slice = seed_frames[slice_index - 1]
        if np.sum(seed_slice) == 0:
            continue  # シード点がないスライスはスキップ
        label, numbers = ndimage.label(seed_slice)
        print(f"スライス {slice_index} のシード点数: {numbers}")  # デバッグ用
        label[label > 0] += seed_index  # ラベル番号をシード点数分ずらす
        seed_frames[slice_index - 1] = label.astype(np.int16)
        seed_index += numbers
        # tiff.imwrite(f"./study/Watershed/Data/Input/20260313_test/seed_slice_{slice_index}.tif", seed_frames[slice_index])
    print(f"総シード点数: {seed_index}")  # デバッグ用
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, seed_frames.astype(np.int16))

    if False:
        centroid_output_path = os.path.splitext(output_path)[0] + "_centroid.tif"
        centroid_volume = _build_seed_centroid_volume(seed_frames.astype(np.int16))
        tiff.imwrite(centroid_output_path, centroid_volume)
        print(f"重心可視化データを保存: {centroid_output_path}")
