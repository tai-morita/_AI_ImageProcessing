from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi


def trim_leading_trailing_zero_slices(volume: np.ndarray, keep_frames: int = 1) -> np.ndarray:
    # 先頭・末尾の連続ゼロスライスを削除しつつ、前後にkeep_frames分を残す。
    nonzero_mask = np.any(volume != 0, axis=(1, 2))
    nonzero_indices = np.where(nonzero_mask)[0]
    if nonzero_indices.size == 0:
        # 全スライスがゼロの場合は元のshapeを保ったまま返す。
        return volume.copy()

    keep_frames = max(0, int(keep_frames))
    start = max(0, int(nonzero_indices[0]) - keep_frames)
    end = min(volume.shape[0], int(nonzero_indices[-1]) + 1 + keep_frames)
    return volume[start:end]

def edit_array(input_path: str) -> np.ndarray:
    # 配列を編集する。
    volume = tiff.imread(input_path)
    new_volume = np.zeros_like(volume)
    for i, slice in enumerate(volume):
        mean_slice = np.mean(slice)
        if mean_slice > 0:
            new_volume[i] = slice
    # 前後1フレームを残してゼロスライスを削除して返す。
    return trim_leading_trailing_zero_slices(new_volume, keep_frames=1)

def cut_slice_range(volume: np.ndarray, start_slice: int, end_slice: int) -> np.ndarray:
    # 指定されたスライス範囲を切り出す。0-basedのスライス番号を使用。
    if start_slice < 0 or start_slice > end_slice:
        raise ValueError(f"Invalid slice range: {start_slice}-{end_slice}")
    if end_slice > volume.shape[0]:
        print(f"Warning: end_slice {end_slice} exceeds volume depth {volume.shape[0]}. Clamping to max depth.")
        end_slice = volume.shape[0]
    return volume[start_slice:end_slice]

def _extract_boundary_points(mask_2d: np.ndarray) -> np.ndarray:
    """2値マスクから外周点を抽出する。

    引数:
        mask_2d: 2次元の2値マスク。
    返り値:
        shape=(N, 2) の外周点配列。列は[x, y]。
    """
    eroded = ndi.binary_erosion(mask_2d, structure=np.ones((3, 3), dtype=bool))
    boundary = mask_2d & (~eroded)
    ys, xs = np.where(boundary)
    return np.column_stack([xs, ys]).astype(np.float64)


def _fit_ellipse_pca(points_xy: np.ndarray) -> dict[str, float]:
    """点群にPCAベースで楕円を近似する。

    引数:
        points_xy: shape=(N, 2) の点群配列。列は[x, y]。
    返り値:
        中心座標・長短半径・回転角(度)を含む辞書。
    """
    if points_xy.shape[0] < 5:
        raise ValueError("楕円近似には5点以上が必要です。")

    center = points_xy.mean(axis=0)
    centered = points_xy - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_radius = 2.0 * math.sqrt(max(float(eigvals[0]), 1e-12))
    minor_radius = 2.0 * math.sqrt(max(float(eigvals[1]), 1e-12))
    angle_rad = math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
        "angle_deg": float(np.degrees(angle_rad)),
    }


def _fit_ellipse_pca_weighted_by_distance(
    projection_mask: np.ndarray,
    boundary_points: np.ndarray,
    distance_gamma: float = 1.5,
) -> dict[str, float]:
    """距離画像重み付きで楕円を近似する。

    引数:
        projection_mask: 成分の2D投影マスク。
        boundary_points: shape=(N, 2) の外周点配列。列は[x, y]。
        distance_gamma: 距離重みの指数。大きいほど中心部を重視する。
    返り値:
        中心座標・長短半径・回転角(度)を含む辞書。
    """
    if boundary_points.shape[0] < 5:
        raise ValueError("楕円近似には5点以上が必要です。")

    if distance_gamma <= 0:
        raise ValueError("distance_gammaは0より大きい値を指定してください。")

    # 内部距離を重みとして使い、距離の遠い(中心寄りの)画素ほど強く寄与させる。
    dist_inside = ndi.distance_transform_edt(projection_mask)
    ys, xs = np.where(projection_mask)
    points_inside = np.column_stack([xs, ys]).astype(np.float64)

    if points_inside.shape[0] < 5:
        return _fit_ellipse_pca(boundary_points)

    weights = np.power(dist_inside[ys, xs].astype(np.float64), distance_gamma)
    weights = np.maximum(weights, 1e-12)
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return _fit_ellipse_pca(boundary_points)

    center = np.sum(points_inside * weights[:, None], axis=0) / w_sum
    centered = points_inside - center
    cov = (centered * weights[:, None]).T @ centered / w_sum

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis_major = eigvecs[:, 0]
    axis_minor = eigvecs[:, 1]
    angle_rad = math.atan2(float(axis_major[1]), float(axis_major[0]))

    # 半径は境界点を主軸へ射影し、外れ値に強い分位値で推定する。
    centered_boundary = boundary_points - center
    proj_major = np.abs(centered_boundary @ axis_major)
    proj_minor = np.abs(centered_boundary @ axis_minor)
    major_radius = float(np.percentile(proj_major, 95))
    minor_radius = float(np.percentile(proj_minor, 95))

    major_radius = max(major_radius, 1e-6)
    minor_radius = max(minor_radius, 1e-6)

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
        "angle_deg": float(np.degrees(angle_rad)),
    }


def _make_filled_ellipse_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    major_radius: float,
    minor_radius: float,
    angle_deg: float,
) -> np.ndarray:
    """指定パラメータの塗りつぶし楕円マスクを生成する。

    引数:
        shape: 出力マスクの形状(height, width)。
        center_x: 楕円中心のx座標。
        center_y: 楕円中心のy座標。
        major_radius: 長半径。
        minor_radius: 短半径。
        angle_deg: 回転角(度)。
    返り値:
        shapeと同サイズの2値マスク。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]

    ang = np.deg2rad(angle_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    x = xx - center_x
    y = yy - center_y

    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a

    major_radius = max(float(major_radius), 1e-6)
    minor_radius = max(float(minor_radius), 1e-6)
    inside = (xr / major_radius) ** 2 + (yr / minor_radius) ** 2 <= 1.0
    return inside


def _plot_component_overlay(
    projection_mask: np.ndarray,
    boundary_points: np.ndarray,
    ellipse_boundary: np.ndarray,
    filled_mask: np.ndarray,
    save_path: Path,
    component_id: int,
) -> None:
    """外周点・楕円境界・塗りつぶし領域を重ね描画して保存する。

    引数:
        projection_mask: 成分の2D投影マスク。
        boundary_points: 外周点群。
        ellipse_boundary: 楕円境界点群。
        filled_mask: 塗りつぶし楕円マスク。
        save_path: 出力画像パス。
        component_id: 成分ID。
    返り値:
        なし。
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(projection_mask, cmap="gray")

    overlay = np.zeros((*filled_mask.shape, 4), dtype=np.float32)
    overlay[filled_mask] = [0.2, 0.8, 0.2, 0.35]
    ax.imshow(overlay)

    ax.plot(boundary_points[:, 0], boundary_points[:, 1], ".", ms=1.5, label="boundary")
    ax.plot(ellipse_boundary[:, 0], ellipse_boundary[:, 1], "r-", lw=1.3, label="ellipse")
    ax.set_title(f"Component {component_id}")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_component_process(
    projection_mask: np.ndarray,
    boundary_points: np.ndarray,
    ellipse_boundary: np.ndarray,
    filled_mask: np.ndarray,
    component_slice_mask: np.ndarray,
    output_slice_mask: np.ndarray,
    save_path: Path,
    component_id: int,
) -> None:
    """fit_circleの処理ステップを6パネルで可視化して保存する。

    引数:
        projection_mask: 成分の2D投影マスク。
        boundary_points: 外周点群。
        ellipse_boundary: 楕円境界点群。
        filled_mask: 縮小後の塗りつぶし楕円マスク。
        component_slice_mask: 元成分の代表スライスマスク。
        output_slice_mask: 出力seedの代表スライスマスク。
        save_path: 出力画像パス。
        component_id: 成分ID。
    返り値:
        なし。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axs = axes.ravel()

    axs[0].imshow(projection_mask, cmap="gray")
    axs[0].set_title("1) Projection mask")

    axs[1].imshow(projection_mask, cmap="gray")
    axs[1].plot(boundary_points[:, 0], boundary_points[:, 1], "c.", ms=1.2)
    axs[1].set_title("2) Boundary points")

    axs[2].imshow(projection_mask, cmap="gray")
    axs[2].plot(ellipse_boundary[:, 0], ellipse_boundary[:, 1], "r-", lw=1.5)
    axs[2].set_title("3) Fitted ellipse")

    axs[3].imshow(filled_mask, cmap="Greens")
    axs[3].set_title("4) Shrunk filled ellipse")

    axs[4].imshow(component_slice_mask, cmap="gray")
    axs[4].set_title("5) Original component (mid-z)")

    axs[5].imshow(output_slice_mask, cmap="gray")
    axs[5].set_title("6) Output seed (mid-z)")

    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"fit_circle process: component {component_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _ellipse_boundary_points(
    center_x: float,
    center_y: float,
    major_radius: float,
    minor_radius: float,
    angle_deg: float,
    n_points: int = 360,
) -> np.ndarray:
    """描画用の楕円境界点を生成する。

    引数:
        center_x: 楕円中心のx座標。
        center_y: 楕円中心のy座標。
        major_radius: 長半径。
        minor_radius: 短半径。
        angle_deg: 回転角(度)。
        n_points: 境界サンプリング点数。
    返り値:
        shape=(n_points, 2) の境界点配列。
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)

    ang = np.deg2rad(angle_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    x = center_x + major_radius * ct * cos_a - minor_radius * st * sin_a
    y = center_y + major_radius * ct * sin_a + minor_radius * st * cos_a
    return np.column_stack([x, y])


def fit_circle(
    input_path: str,
    shrink_ratio: float = 0.92,
    output_path: str | None = None,
    save_process_figure: bool = True,
    use_distance_weighted_fit: bool = True,
    distance_gamma: float = 1.5,
) -> np.ndarray:
    """各スライスのConnected Componentごとに楕円内側を塗って縮小seedを作る。

    引数:
        input_path: 入力seedボリューム(tif)のパス。
        shrink_ratio: 楕円半径の縮小率(0より大、1以下推奨)。
        output_path: 出力seedボリュームの保存先。Noneなら自動命名。
        save_process_figure: Trueなら処理ステップ可視化画像を保存する。
        use_distance_weighted_fit: Trueなら距離画像重み付きで楕円近似する。
        distance_gamma: 距離重みの指数。大きいほど中心部を重視する。
    返り値:
        楕円内側塗りつぶし後の3Dラベル配列。
    """
    if shrink_ratio <= 0:
        raise ValueError("shrink_ratioは0より大きい値を指定してください。")

    volume = tiff.imread(input_path)
    binary = volume > 0

    if not np.any(binary):
        raise ValueError("入力ボリュームに有効なseedがありません。")

    input_path_obj = Path(input_path)
    out_dir = input_path_obj.parent / f"{input_path_obj.stem}_ellipse_shrunk"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_seed = np.zeros(binary.shape, dtype=np.int32)
    params_list: list[dict[str, float | int]] = []
    output_label = 1

    for z in range(binary.shape[0]):
        slice_mask = binary[z]
        if not np.any(slice_mask):
            continue

        labeled_slice, num_components = ndi.label(slice_mask, structure=np.ones((3, 3), dtype=np.uint8))
        for slice_component_id in range(1, num_components + 1):
            component_slice_mask = labeled_slice == slice_component_id
            if not np.any(component_slice_mask):
                continue

            boundary_points = _extract_boundary_points(component_slice_mask)
            if boundary_points.shape[0] < 5:
                continue

            if use_distance_weighted_fit:
                ellipse = _fit_ellipse_pca_weighted_by_distance(
                    projection_mask=component_slice_mask,
                    boundary_points=boundary_points,
                    distance_gamma=distance_gamma,
                )
            else:
                ellipse = _fit_ellipse_pca(boundary_points)

            ellipse["major_radius"] = float(ellipse["major_radius"] * shrink_ratio)
            ellipse["minor_radius"] = float(ellipse["minor_radius"] * shrink_ratio)

            filled_mask = _make_filled_ellipse_mask(
                shape=component_slice_mask.shape,
                center_x=float(ellipse["center_x"]),
                center_y=float(ellipse["center_y"]),
                major_radius=float(ellipse["major_radius"]),
                minor_radius=float(ellipse["minor_radius"]),
                angle_deg=float(ellipse["angle_deg"]),
            )

            component_id = output_label
            output_seed[z][filled_mask] = component_id

            ellipse_boundary = _ellipse_boundary_points(
                center_x=float(ellipse["center_x"]),
                center_y=float(ellipse["center_y"]),
                major_radius=float(ellipse["major_radius"]),
                minor_radius=float(ellipse["minor_radius"]),
                angle_deg=float(ellipse["angle_deg"]),
            )
            plot_path = out_dir / f"z{z:04d}_component_{component_id:04d}.png"
            _plot_component_overlay(
                projection_mask=component_slice_mask,
                boundary_points=boundary_points,
                ellipse_boundary=ellipse_boundary,
                filled_mask=filled_mask,
                save_path=plot_path,
                component_id=component_id,
            )

            if save_process_figure:
                output_slice_mask = output_seed[z] == component_id
                process_path = out_dir / f"z{z:04d}_component_{component_id:04d}_process.png"
                _plot_component_process(
                    projection_mask=component_slice_mask,
                    boundary_points=boundary_points,
                    ellipse_boundary=ellipse_boundary,
                    filled_mask=filled_mask,
                    component_slice_mask=component_slice_mask,
                    output_slice_mask=output_slice_mask,
                    save_path=process_path,
                    component_id=component_id,
                )

            params_list.append(
                {
                    "component_id": component_id,
                    "slice_index": int(z),
                    "slice_component_id": int(slice_component_id),
                    "boundary_points": int(boundary_points.shape[0]),
                    "voxel_count_original": int(np.sum(component_slice_mask)),
                    "center_x": float(ellipse["center_x"]),
                    "center_y": float(ellipse["center_y"]),
                    "major_radius": float(ellipse["major_radius"]),
                    "minor_radius": float(ellipse["minor_radius"]),
                    "angle_deg": float(ellipse["angle_deg"]),
                    "shrink_ratio": float(shrink_ratio),
                    "use_distance_weighted_fit": bool(use_distance_weighted_fit),
                    "distance_gamma": float(distance_gamma),
                }
            )
            output_label += 1

    if output_path is None:
        output_path = str(out_dir / f"{input_path_obj.stem}_seed_shrunk_ellipse.tif")

    tiff.imwrite(output_path, output_seed.astype(np.int32))

    json_path = out_dir / "ellipse_params.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(params_list, f, ensure_ascii=False, indent=2)

    return output_seed

def trim_by_top50_far_region(mask_2d: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """
    mask_2d: 2D二値マスク（True/1: 対象領域）
    keep_ratio: 距離の大きい側から残す割合（0.5なら上位50%）
    戻り値: トリミング後の2D二値マスク
    """
    if mask_2d.ndim != 2:
        raise ValueError("mask_2d must be 2D.")
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    mask = mask_2d.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    # 内部から境界までの距離
    dist = ndi.distance_transform_edt(mask)

    # 対象領域内の距離だけ見る
    vals = dist[mask]
    # keep_ratio=0.5 -> 50パーセンタイル以上を残す
    q = 100.0 * (1.0 - keep_ratio)
    th = np.percentile(vals, q)

    trimmed = mask & (dist >= th)
    return trimmed


def trim_volume_by_top50_far_region(volume_3d: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """3Dボリュームをスライスごとに2D距離ベースでトリミングする。

    引数:
        volume_3d: 3D入力ボリューム。
        keep_ratio: 距離の大きい側から残す割合。
    返り値:
        各スライスでトリミングした3D二値マスク。
    """
    if volume_3d.ndim != 3:
        raise ValueError("volume_3d must be 3D.")

    trimmed = np.zeros(volume_3d.shape, dtype=bool)
    for z in range(volume_3d.shape[0]):
        trimmed[z] = trim_by_top50_far_region(volume_3d[z], keep_ratio=keep_ratio)
    return trimmed


def relabel_trimmed_components_per_slice(trimmed_mask_3d: np.ndarray) -> np.ndarray:
    """トリミング後マスクをスライスごとに連結成分ラベリングし直す。

    引数:
        trimmed_mask_3d: 3Dの二値マスク。
    返り値:
        スライスごとの2D連結成分を、全体で連番にした3Dラベル画像。
    """
    if trimmed_mask_3d.ndim != 3:
        raise ValueError("trimmed_mask_3d must be 3D.")

    relabeled = np.zeros(trimmed_mask_3d.shape, dtype=np.int32)
    next_label = 1

    for z in range(trimmed_mask_3d.shape[0]):
        labeled_slice, num_components = ndi.label(
            trimmed_mask_3d[z].astype(bool),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if num_components == 0:
            continue

        nonzero = labeled_slice > 0
        labeled_slice = labeled_slice.astype(np.int32)
        labeled_slice[nonzero] += next_label - 1
        relabeled[z] = labeled_slice
        next_label += int(num_components)

    return relabeled

def average_slice():
    """3Dボリュームのスライス方向をつぶす

    引数:
        volume: 3D入力ボリューム。
    返り値:
        平均化された2D画像。
    """
    input_paths = [r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\lower_jaw\lower_jaw_seed_before_edit.tif",
                   r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\lower_jaw\lower_jaw_seed_after_edit.tif",
                   r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\lower_jaw\lower_jaw_trimmed_volume.tif"]
    for path in input_paths:
        volume = tiff.imread(path)
        if volume.ndim != 3:
            raise ValueError("volume must be 3D.")
        average_image = np.mean(volume, axis=0)
        output_path = path.replace(".tif", "_average.tif")
        tiff.imwrite(output_path, average_image.astype(np.float32))
    return average_image

if __name__ == "__main__":
    r"""
    input_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\_label_map_8_filled_aligned.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume.tif"
    edited_volume = edit_array(input_path)
    tiff.imwrite(output_path, edited_volume.astype(np.float32))
    input_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\lower_jaw.tif"
    cut_volume = cut_slice_range(tiff.imread(input_path), start_slice=0, end_slice=237)
    tiff.imwrite(output_path, cut_volume.astype(np.float32))
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\upper_jaw.tif"
    cut_volume = cut_slice_range(tiff.imread(input_path), start_slice=154, end_slice=10000)
    tiff.imwrite(output_path, cut_volume.astype(np.float32))
    fit_circle(
        input_path=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume.tif",
        shrink_ratio=0.5,
        output_path=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume_fitting.tif",
        save_process_figure=True,
        use_distance_weighted_fit=True,
        distance_gamma=2.0,
    )

    input_path  = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume_trimmed.tif"

    volume = tiff.imread(input_path)
    trimmed_mask = trim_volume_by_top50_far_region(volume, keep_ratio=0.25)
    trimmed_labeled = relabel_trimmed_components_per_slice(trimmed_mask)
    tiff.imwrite(output_path, trimmed_labeled.astype(np.uint16))
    """
    average_slice()