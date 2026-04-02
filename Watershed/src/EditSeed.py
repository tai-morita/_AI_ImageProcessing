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


def _normalize_targets(
    coordinate_yx: tuple[int, int] | list[tuple[int, int]],
    labeling_no: int | list[int],
) -> list[tuple[tuple[int, int], int]]:
    if isinstance(coordinate_yx, tuple) and isinstance(labeling_no, int):
        return [(coordinate_yx, labeling_no)]

    if not isinstance(coordinate_yx, list) or not isinstance(labeling_no, list):
        raise ValueError("coordinate_yx and labeling_no must both be single values or both be lists.")

    if len(coordinate_yx) != len(labeling_no):
        raise ValueError("coordinate_yx and labeling_no must have the same length.")

    targets: list[tuple[tuple[int, int], int]] = []
    for coord, label in zip(coordinate_yx, labeling_no):
        if len(coord) != 2:
            raise ValueError(f"invalid coordinate: {coord}")
        if label <= 0:
            raise ValueError("labeling_no must be a positive integer.")
        targets.append(((int(coord[0]), int(coord[1])), int(label)))

    return targets


def _normalize_slice_ranges(
    slice_ranges: tuple[int, int] | list[tuple[int, int]] | None,
    n_targets: int,
    n_slices: int,
) -> list[tuple[int, int]]:
    if slice_ranges is None:
        return [(0, n_slices - 1)] * n_targets

    if isinstance(slice_ranges, tuple):
        ranges = [slice_ranges] * n_targets
    elif isinstance(slice_ranges, list):
        if len(slice_ranges) != n_targets:
            raise ValueError("slice_ranges length must match number of targets.")
        ranges = slice_ranges
    else:
        raise ValueError("slice_ranges must be None, tuple, or list of tuples.")

    normalized: list[tuple[int, int]] = []
    for start_slice, end_slice in ranges:
        start_slice = int(start_slice)
        end_slice = int(end_slice)
        if not (0 <= start_slice <= end_slice < n_slices):
            raise ValueError(
                f"invalid slice range: ({start_slice}, {end_slice}) for n_slices={n_slices}"
            )
        normalized.append((start_slice, end_slice))

    return normalized


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


def _validate_and_normalize_volume(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise ValueError("input must be 2D or 3D (slice, y, x).")
    return volume


def _extract_component_mask_2d(
    binary_slice: np.ndarray,
    coordinate_yx: tuple[int, int],
    connectivity: int = 1,
) -> np.ndarray:
    y, x = coordinate_yx
    if not binary_slice[y, x]:
        return np.zeros_like(binary_slice, dtype=bool)

    structure = ndimage.generate_binary_structure(rank=2, connectivity=connectivity)
    labeled_slice, _ = ndimage.label(binary_slice, structure=structure)
    target_label = int(labeled_slice[y, x])
    return labeled_slice == target_label


def profile_component_area_by_slice(
    volume_label_path: str,
    coordinate_yx: tuple[int, int],
    connectivity: int = 1,
) -> list[dict[str, int]]:
    """
    指定座標 (y, x) が属する2D connected componentの面積をスライスごとに求める。
    指定座標が背景のスライスは area=0 を返す。
    """
    volume = _validate_and_normalize_volume(tiff.imread(volume_label_path))
    _, height, width = volume.shape

    y, x = coordinate_yx
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"coordinate {coordinate_yx} is out of bounds for shape {volume.shape}.")

    profile: list[dict[str, int]] = []
    for slice_index, slice_image in enumerate(volume):
        component_mask = _extract_component_mask_2d(slice_image > 0, coordinate_yx, connectivity)
        area = int(np.count_nonzero(component_mask))
        profile.append({"slice": int(slice_index), "area": area})

    return profile


def label_previous_slice_on_large_area_diff(
    volume_label_path: str,
    output_path: str,
    coordinate_yx: tuple[int, int] | list[tuple[int, int]],
    labeling_no: int | list[int],
    diff_threshold: int = 1000,
    connectivity: int = 1,
    slice_ranges: tuple[int, int] | list[tuple[int, int]] | None = None,
    mark_coordinates: bool = False,
    coordinate_mark_value: int = 255,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """
    指定座標の connected component 面積をスライスごとに求め、
    差分 diff=current-prev に対して次を行う。
    - diff > +diff_threshold: 前スライス側の connected component を labeling_no でラベリング
    - diff < -diff_threshold: 現スライス側の connected component を labeling_no でラベリング
    最初に閾値到達した時点で、そのターゲットの探索を終了する。

    Returns
    -------
    profile : list[dict[str, object]]
        スライスごとの面積情報。
    labeled_output : np.ndarray
        保存したラベルボリューム。

    Notes
    -----
    mark_coordinates=True の場合、指定座標(y, x)を slice_ranges の範囲内で
    coordinate_mark_value でマーキングしてから保存する。
    """
    original = tiff.imread(volume_label_path)
    volume = _validate_and_normalize_volume(original)
    targets = _normalize_targets(coordinate_yx, labeling_no)
    ranges = _normalize_slice_ranges(slice_ranges, len(targets), volume.shape[0])

    if len(targets) == 1:
        profile: list[dict[str, object]] = profile_component_area_by_slice(
            volume_label_path, targets[0][0], connectivity
        )
    else:
        profile = []

    labeled_output = np.zeros_like(volume, dtype=np.int16)
    binary_volume = volume > 0
    labeled_slice_indices: list[int] = []

    for target_index, (target_coordinate_yx, target_labeling_no) in enumerate(targets):
        print(f"ターゲット {target_index}: coordinate={target_coordinate_yx}, labeling_no={target_labeling_no}")
        start_slice, end_slice = ranges[target_index]
        target_profile = profile_component_area_by_slice(volume_label_path, target_coordinate_yx, connectivity)
        threshold_reached = False
        if len(targets) > 1:
            profile.extend(
                {
                    "target_index": target_index,
                    "coordinate_yx": target_coordinate_yx,
                    "labeling_no": target_labeling_no,
                    "slice": row["slice"],
                    "area": row["area"],
                }
                for row in target_profile
            )

        # 閾値判定は各ターゲットの開始スライスから行う。
        # diffは current-prev なので、最初の比較点は start_slice+1。
        loop_start = max(1, start_slice + 1)
        loop_end = min(end_slice, len(target_profile) - 1)

        for slice_index in range(loop_start, loop_end + 1):
            prev_area = target_profile[slice_index - 1]["area"]
            current_area = target_profile[slice_index]["area"]
            diff = current_area - prev_area
            if -diff_threshold <= diff <= diff_threshold:
                continue

            if diff > diff_threshold:
                target_slice_for_label = slice_index - 1
            else:
                target_slice_for_label = slice_index

            target_component_mask = _extract_component_mask_2d(
                binary_volume[target_slice_for_label], target_coordinate_yx, connectivity
            )
            if np.any(target_component_mask):
                labeled_output[target_slice_for_label][target_component_mask] = target_labeling_no
                labeled_slice_indices.append(target_slice_for_label)
                print(
                    f"ラベリング対象スライス: {target_slice_for_label} "
                    f"coordinate={target_coordinate_yx}, label={target_labeling_no}, "
                    f"range=({start_slice}, {end_slice}), "
                    f"(prev_area={prev_area}, current_area={current_area}, diff={diff})"
                )
                print(
                    f"coordinate={target_coordinate_yx} は最初の閾値到達で終了します: "
                    f"{target_slice_for_label} で diff={diff} が閾値 +/-{diff_threshold} を超えました"
                )
                threshold_reached = True
                break

        # 探索範囲で閾値未到達なら、最大面積スライスをラベリングする。
        if not threshold_reached:
            range_rows = target_profile[start_slice : end_slice + 1]
            if len(range_rows) > 0:
                max_row = max(range_rows, key=lambda row: int(row["area"]))
                max_slice = int(max_row["slice"])
                max_area = int(max_row["area"])
                max_component_mask = _extract_component_mask_2d(
                    binary_volume[max_slice], target_coordinate_yx, connectivity
                )
                if np.any(max_component_mask):
                    labeled_output[max_slice][max_component_mask] = target_labeling_no
                    labeled_slice_indices.append(max_slice)
                    print(
                        f"閾値未到達のため最大面積スライスをラベリング: {max_slice} "
                        f"coordinate={target_coordinate_yx}, label={target_labeling_no}, area={max_area}, "
                        f"range=({start_slice}, {end_slice})"
                    )
                else:
                    print(
                        f"閾値未到達かつ最大面積スライス({max_slice})に前景がないためラベリングなし: "
                        f"coordinate={target_coordinate_yx}, range=({start_slice}, {end_slice})"
                    )

    save_dir = os.path.dirname(output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if len(labeled_slice_indices) == 0:
        print("ラベリング対象スライスはありません")
    else:
        print(f"ラベリングしたスライス番号: {labeled_slice_indices}")

    if mark_coordinates:
        marked_count = 0
        for target_index, (target_coordinate_yx, _) in enumerate(targets):
            y, x = target_coordinate_yx
            start_slice, end_slice = ranges[target_index]
            labeled_output[start_slice : end_slice + 1, y, x] = coordinate_mark_value
            marked_count += end_slice - start_slice + 1
        print(
            f"指定座標マーキングを適用: value={coordinate_mark_value}, total_marked_voxels={marked_count}"
        )

    save_volume = labeled_output[0] if original.ndim == 2 else labeled_output
    tiff.imwrite(output_path, save_volume.astype(np.int16))
    return profile, save_volume.astype(np.int16)


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

def edit_seed_1teeth_per_slice(volume_label_path, volume_label_edit_path, output_path):
    # 1スライスに1歯としてシード点を編集する
    original_frames = tiff.imread(volume_label_path)
    edit_frames = tiff.imread(volume_label_edit_path)
    seed_frames = original_frames - edit_frames
    seed_index = 0  # ラベル番号の開始
    for slice_index in range(1, seed_frames.shape[0]+1):
        seed_slice = seed_frames[slice_index - 1]
        if np.sum(seed_slice) == 0:
            continue  # シード点がないスライスはスキップ
        # 2値化してラベリング。同一スライスは同じラベリングにする
        binary_seed_slice = seed_slice > 0
        label, numbers = ndimage.label(binary_seed_slice)
        print(f"スライス {slice_index} のシード点数: {numbers}")  # デバッグ用
        if numbers > 0:
            seed_frames[slice_index - 1] = (label > 0).astype(np.int16) * (seed_index + 1)
            seed_index += 1

    print(f"総シード点数: {seed_index}")  # デバッグ用
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, seed_frames.astype(np.int16))

    if False:
        centroid_output_path = os.path.splitext(output_path)[0] + "_centroid.tif"
        centroid_volume = _build_seed_centroid_volume(seed_frames.astype(np.int16))
        tiff.imwrite(centroid_output_path, centroid_volume)
        print(f"重心可視化データを保存: {centroid_output_path}")
