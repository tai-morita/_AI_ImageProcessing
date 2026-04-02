import glob
import os

import tifffile as tiff
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt

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