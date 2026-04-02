import glob
import os

import tifffile as tiff
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
try:
    from .AnnotationONGUI import load_3d_tiff, SliceViewer3D
    from .EditSeed import label_previous_slice_on_large_area_diff, profile_component_area_by_slice
except ImportError:
    # Support direct execution: python Watershed/src/temp.py
    from AnnotationONGUI import load_3d_tiff, SliceViewer3D
    from EditSeed import label_previous_slice_on_large_area_diff, profile_component_area_by_slice

def save_dist(input_path):
    # 距離画像に変換する
    # 2D と 3D 両方
    frames = tiff.imread(input_path)
    dist_2d = []
    for frame in frames:
        dist = ndi.distance_transform_edt(frame)
        dist_2d.append(dist)
    tiff.imwrite(os.path.join(os.path.dirname(input_path), 
                              f"{os.path.splitext(os.path.basename(input_path))[0]}_dist_2d.tif"), 
                              np.array(dist_2d).astype(np.float32))
    dist_3d = ndi.distance_transform_edt(frames)
    tiff.imwrite(os.path.join(os.path.dirname(input_path), 
                              f"{os.path.splitext(os.path.basename(input_path))[0]}_dist_3d.tif"), 
                              dist_3d.astype(np.float32))


def keep_top_percent_distance(
    input_path: str,
    top_percent: float,
    save_path: str | None = None,
) -> np.ndarray:
    """
    距離画像のうち、上位 top_percent [%] の画素だけを True で返す。
    判定は全ボリュームの値で行う。
    """
    distance = tiff.imread(input_path)
    if not (0 < top_percent <= 100):
        raise ValueError("top_percent must be in (0, 100].")

    values = distance[distance > 0]
    if values.size == 0:
        return np.zeros_like(distance, dtype=bool)

    # 上位X%に対応する下限値
    threshold = np.percentile(values, 100.0 - top_percent)

    out = np.zeros_like(distance, dtype=bool)
    out[distance > 0] = distance[distance > 0] >= threshold

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tiff.imwrite(save_path, out.astype("uint8") * 255)
    return out


def profile_component_area_by_frame(
    input_path: str,
    coordinate_yx: tuple[int, int],
    connectivity: int = 1,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[dict[str, int]]:
    """
    指定画素 (y, x) について、各フレームでその画素が属する
    2D connected component の面積を返す。

    入力は 3D (frame, y, x) を想定。2D入力なら1フレームとして扱う。
    画素が背景(0)のフレームは area=0 とする。
    connectivity は 1=4近傍, 2=8近傍。
    start_frame, end_frame は探索するフレーム範囲（両端含む）。
    """
    volume = tiff.imread(input_path)
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise ValueError("input must be 2D or 3D (frame, y, x).")

    y, x = coordinate_yx
    n_frames, y_max, x_max = volume.shape
    if not (0 <= y < y_max and 0 <= x < x_max):
        raise ValueError(f"coordinate {coordinate_yx} is out of bounds for shape {volume.shape}.")

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames - 1
    if not (0 <= start_frame <= end_frame < n_frames):
        raise ValueError(
            f"invalid frame range: start_frame={start_frame}, end_frame={end_frame}, n_frames={n_frames}"
        )

    structure = ndi.generate_binary_structure(rank=2, connectivity=connectivity)
    profile: list[dict[str, int]] = []

    for fi in range(start_frame, end_frame + 1):
        frame = volume[fi]
        mask = frame > 0
        if not mask[y, x]:
            profile.append({"frame": int(fi), "area": 0})
            continue

        labeled_cc, _ = ndi.label(mask, structure=structure)
        target_label = int(labeled_cc[y, x])
        area = int(np.count_nonzero(labeled_cc == target_label))
        profile.append({"frame": int(fi), "area": area})

    return profile


def set_component_value_by_frame_and_save(
    input_path: str,
    output_path: str,
    coordinate_yx: tuple[int, int],
    value: int = 100,
    connectivity: int = 1,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> np.ndarray:
    """
    指定画素 (y, x) を含む 2D connected component のみ、各フレームで value に変更して保存する。
    入力は 3D (frame, y, x) を想定。2D入力なら1フレームとして扱う。
    """
    original = tiff.imread(input_path)
    was_2d = original.ndim == 2

    if was_2d:
        volume = original[np.newaxis, ...]
    else:
        volume = original

    if volume.ndim != 3:
        raise ValueError("input must be 2D or 3D (frame, y, x).")

    out = volume.copy()
    if out.dtype == np.bool_:
        out = out.astype(np.uint8)

    y, x = coordinate_yx
    n_frames, y_max, x_max = volume.shape
    if not (0 <= y < y_max and 0 <= x < x_max):
        raise ValueError(f"coordinate {coordinate_yx} is out of bounds for shape {volume.shape}.")

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames - 1
    if not (0 <= start_frame <= end_frame < n_frames):
        raise ValueError(
            f"invalid frame range: start_frame={start_frame}, end_frame={end_frame}, n_frames={n_frames}"
        )

    structure = ndi.generate_binary_structure(rank=2, connectivity=connectivity)

    for fi in range(start_frame, end_frame + 1):
        frame = volume[fi]
        mask = frame > 0
        if not mask[y, x]:
            continue

        labeled_cc, _ = ndi.label(mask, structure=structure)
        target_label = int(labeled_cc[y, x])
        target_mask = labeled_cc == target_label
        out[fi][target_mask] = value

    output = out[0] if was_2d else out
    save_dir = os.path.dirname(output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tiff.imwrite(output_path, output)
    return output


def show_component_profile(profile: list[dict[str, int]], title: str = "Component area by frame") -> None:
    """
    profile_component_area_by_frame の結果をプロットして表示する。
    """
    if len(profile) == 0:
        print("empty profile")
        return

    zs = [int(row["frame"]) for row in profile]
    areas = [int(row["area"]) for row in profile]
    d_areas = np.diff(areas)
    d_frames = zs[1:]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    line1 = ax1.plot(zs, areas, marker="o", linewidth=0.8, color="tab:blue", label="area")
    ax1.set_title(title)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("area (pixels)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(d_frames, d_areas, marker="x", linestyle="--", linewidth=0.8, color="tab:red", label="d(area)")
    ax2.set_ylabel("d(area)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    plt.show()


def moving_average_1d(array, window=5, mode="same"):
    """
    1次元配列の移動平均
    mode="same" なら元と同じ長さで返す
    """
    a = np.asarray(array, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(array, kernel, mode=mode)

def get_profile(file_path: str) -> list[dict[int]]:
    # file_path のプロファイルを取得する
    # 差分が負から正に変わるスライスを探す
    frames = tiff.imread(file_path)
    profile = []
    for slice_index in range(frames.shape[0]):
        area = np.count_nonzero(frames[slice_index])
        profile.append({"slice": slice_index, "area": area})
    # 移動平均して均してから極小値を求める
    profile_smoothed = moving_average_1d([row["area"] for row in profile], window=5)
    difference = np.diff(profile_smoothed)
    extrema_indices = np.where((difference[:-1] < 0) & (difference[1:] > 0))[0] + 1
    extrema_slices = [profile[idx]["slice"] for idx in extrema_indices]
    extrema_areas = [profile[idx]["area"] for idx in extrema_indices]
    print("Extrema (slice, area):")
    for slice, area in zip(extrema_slices, extrema_areas):
        print(f"Slice: {slice}, Area: {area}")
    return extrema_slices

def auto_annotation(file_path: str, cmap: str = "gray", undo_radius_px: float = 5.0) -> list:
    # スライス画像を手動でマーキングして、Seed画像を作成する
    # 読み込み & 座標の保存はAnnotationONGUI.pyに任せる
    # 座標を label_previous_slice_on_large_area_diff の 引数に渡す
    # 探索するスライスは、 file_path のプロファイルを使う

    volume = load_3d_tiff(file_path)
    viewer = SliceViewer3D(volume=volume, cmap=cmap, undo_radius_px=undo_radius_px)
    viewer.show()
    viewer.print_saved_points()

    extrema_slices = get_profile(file_path)
    if len(viewer.clicked_points) == 0:
        print("No seed points were selected.")
        return []

    # viewer.clicked_points: クリックした点のリスト (slice_index_1based, y, x)
    seed_coordinates = []
    seed_slices = []
    for slice_index_1based, y, x in viewer.clicked_points:
        # クリックした座標が、プロファイルしたデータのうちどのやまにいるかを判定する
        # とりあえず二分探索で
        min_index = 0
        search_slice = [0, volume.shape[0] - 1]
        for target_slice in extrema_slices:
            if slice_index_1based <= target_slice:
                search_slice = [min_index, target_slice]
                break
            else:
                min_index = target_slice
        seed_coordinates.append((y, x))
        seed_slices.append(tuple(search_slice))
    for (y, x), (start_slice, end_slice) in zip(seed_coordinates, seed_slices):
        print(f"Seed coordinate (y, x): ({y}, {x}), search slice range: {start_slice}-{end_slice}")
    labeling_no = [i + 1 for i in range(len(seed_coordinates))]
    output_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}_auto_seed.tif",
    )
    profile, labeled = label_previous_slice_on_large_area_diff(
        volume_label_path=file_path,
        output_path=output_path,
        coordinate_yx=seed_coordinates,
        labeling_no=labeling_no,
        slice_ranges=seed_slices,
        diff_threshold=1000,
        mark_coordinates=True,
        coordinate_mark_value=255,
    )
    watershed_3d_tiff(
        input_path=file_path, 
        volume_label_annotate=labeled,
        output_path=output_path,
        connectivity=1)
    print(f"saved: {output_path}")
    return profile


if __name__ == "__main__":
	file_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1\label_map_1_filled.tif"
	profile = auto_annotation(file_path)

r"""
指定した点に対して、スライスごとにその点が属する連結成分の面積をプロファイルとして取得する。
if __name__ == "__main__":
    input_dir = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1"
    output_dir = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1"

    input_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1\label_map_1_filled.tif"
    # save_dist(input_path)

    coordinate_yx = (289, 141)
    start_frame = 0
    end_frame = 60

    profile = profile_component_area_by_frame(
        input_path=input_path,
        coordinate_yx=coordinate_yx,
        connectivity=1,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    saved_path = os.path.join(output_dir, "label_map_1_component_value100.tif")
    set_component_value_by_frame_and_save(
        input_path=input_path,
        output_path=saved_path,
        coordinate_yx=coordinate_yx,
        value=100,
        connectivity=1,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    print(f"target pixel (y, x): {coordinate_yx}")
    print(f"frame range: {start_frame}-{end_frame}")
    print(f"profile frames: {len(profile)}")
    print(f"saved: {saved_path}")
    show_component_profile(
        profile,
        title=f"Component area profile @ pixel {coordinate_yx}, frames {start_frame}-{end_frame}",
    )
"""