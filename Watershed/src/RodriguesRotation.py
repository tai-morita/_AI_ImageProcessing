import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

# 咬合平面の法線を求めて、スライス面に平行化する


def rodrigues_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """from_vec を to_vec に回す 3x3 回転行列を返す。"""
    a = from_vec / (np.linalg.norm(from_vec) + 1e-12)
    b = to_vec / (np.linalg.norm(to_vec) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        # 平行 or 逆平行
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180度回転: a と直交する軸を1つ作る
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis) + 1e-12
        # 180度回転行列: R = -I + 2uu^T
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    k = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3) + k + (k @ k) * ((1.0 - c) / (s * s))
    return r


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """2値マスクの表面ボクセルを抽出。"""
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)  # 6近傍
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    surface = mask & (~eroded)
    return surface


def choose_occlusal_candidates(
    surface_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    percentile: float,
    side: str,
) -> np.ndarray:
    """
    咬合面候補点を抽出。
    side:
      - high: z が大きい側を候補
      - low : z が小さい側を候補
      - auto: 点数が多い側を採用
    """
    idx = np.argwhere(surface_zyx)
    if idx.size == 0:
        raise ValueError("表面ボクセルが見つかりません。入力が空の可能性があります。")

    z = idx[:, 0]

    z_low_thr = np.percentile(z, percentile)  # 例: 85%
    z_high_thr = np.percentile(z, 100.0 - percentile)  # 例: 15%

    cand_high = idx[z >= z_low_thr]
    cand_low = idx[z <= z_high_thr]

    if side == "high":
        chosen = cand_high
    elif side == "low":
        chosen = cand_low
    else:
        chosen = cand_high if len(cand_high) >= len(cand_low) else cand_low

    if len(chosen) < 20:
        raise ValueError(
            f"咬合面候補点が少なすぎます: {len(chosen)} 点。"
            " percentile を下げるか side を変えてください。"
        )

    # 物理座標へ変換 (z,y,x) * spacing
    pts_mm = chosen.astype(np.float64) * spacing_zyx[None, :]
    return pts_mm


def fit_plane_normal_svd(points_mm: np.ndarray) -> np.ndarray:
    """SVDで平面法線（最小特異ベクトル）を求める。"""
    center = points_mm.mean(axis=0)
    x = points_mm - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal) + 1e-12
    return normal


def rotate_volume_to_occlusal_parallel(
    volume: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    percentile=85.0,
    side="auto",
    pad=20,
    inplane_k45=0,
):
    """
    歯2値ボリュームを回転し、咬合面法線をスライス法線(z軸)に一致させる。
    戻り値:
      rotated_uint8, info(dict)
    """
    if volume.ndim != 3:
        raise ValueError(f"3次元配列を想定していますが shape={volume.shape}")

    mask = volume > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("前景が空です。2値化結果を確認してください。")

    # 端の切れを減らすためにパディング
    if pad > 0:
        mask = np.pad(mask, ((pad, pad), (pad, pad), (pad, pad)), mode="constant")
    shape = np.array(mask.shape, dtype=np.float64)

    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    if np.any(spacing_zyx <= 0):
        raise ValueError(f"spacing は正数である必要があります: {spacing_zyx}")

    surface = extract_surface(mask)
    pts_mm = choose_occlusal_candidates(surface, spacing_zyx, percentile, side)
    n = fit_plane_normal_svd(pts_mm)

    # 方向を +z 側へそろえる（向き反転を防ぐ）
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # z,y,x 系で +z
    if np.dot(n, target) < 0:
        n = -n

    # 物理座標系の回転
    r_phys = rodrigues_rotation_matrix(n, target)

    # index座標系への変換: A = S^-1 R S
    s = np.diag(spacing_zyx)
    s_inv = np.diag(1.0 / spacing_zyx)
    a = s_inv @ r_phys @ s

    # affine_transform は output -> input 変換を受け取るため逆行列を使う
    m = np.linalg.inv(a)
    center = (shape - 1.0) / 2.0
    offset = center - m @ center

    # 2値なので最近傍補間
    rotated = ndi.affine_transform(
        mask.astype(np.uint8),
        matrix=m,
        offset=offset,
        output_shape=tuple(mask.shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    # スライス面内の追加回転（45度刻み）
    inplane_angle_deg = float(inplane_k45) * 45.0
    if abs(inplane_angle_deg) > 1e-12:
        rotated = ndi.rotate(
            rotated,
            angle=inplane_angle_deg,
            axes=(1, 2),
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )

    rotated = (rotated > 0).astype(np.uint8) * 255

    # 参考情報
    cos_theta = float(np.clip(np.dot(n, target), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))

    info = {
        "input_shape_zyx": [int(v) for v in volume.shape],
        "working_shape_zyx": [int(v) for v in mask.shape],
        "spacing_zyx": [float(v) for v in spacing_zyx],
        "estimated_normal_zyx": [float(v) for v in n],
        "target_normal_zyx": [1.0, 0.0, 0.0],
        "rotation_angle_deg": theta_deg,
        "percentile": float(percentile),
        "side": side,
        "pad": int(pad),
        "inplane_k45": int(inplane_k45),
        "inplane_angle_deg": inplane_angle_deg,
    }
    return rotated, info


def estimate_left_right_first_slice_tilt(
    volume: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    first_side="low",
    min_voxels_per_x=1,
    trim_percentile=10.0,
):
    """
    x位置ごとの初出スライスを使い、左右方向のz傾きを推定する。
    first_side:
      - low : z=0側から見て最初に歯が出る位置を使う
      - high: z最大側から見て最初に歯が出る位置を使う
    """
    if volume.ndim != 3:
        raise ValueError(f"3次元配列を想定していますが shape={volume.shape}")

    mask = volume > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("前景が空です。2値化結果を確認してください。")

    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    if np.any(spacing_zyx <= 0):
        raise ValueError(f"spacing は正数である必要があります: {spacing_zyx}")

    z_size, _, x_size = mask.shape
    first_z_values = []
    x_values = []
    projection_zx = np.any(mask, axis=1)
    voxel_counts_per_x = np.count_nonzero(mask, axis=(0, 1))

    for x_index in range(x_size):
        if voxel_counts_per_x[x_index] < min_voxels_per_x:
            continue
        z_has_foreground = projection_zx[:, x_index]
        if not np.any(z_has_foreground):
            continue
        if first_side == "low":
            z_index = int(np.argmax(z_has_foreground))
        elif first_side == "high":
            z_index = int(z_size - 1 - np.argmax(z_has_foreground[::-1]))
        else:
            raise ValueError(f"first_side は 'low' または 'high' を指定してください: {first_side}")
        x_values.append(x_index)
        first_z_values.append(z_index)

    if len(x_values) < 10:
        raise ValueError(f"左右傾き推定に使えるx列が少なすぎます: {len(x_values)}")

    x_values = np.asarray(x_values, dtype=np.float64)
    first_z_values = np.asarray(first_z_values, dtype=np.float64)

    if trim_percentile > 0:
        low_thr = np.percentile(first_z_values, trim_percentile)
        high_thr = np.percentile(first_z_values, 100.0 - trim_percentile)
        keep = (low_thr <= first_z_values) & (first_z_values <= high_thr)
        if np.count_nonzero(keep) >= 10:
            x_values = x_values[keep]
            first_z_values = first_z_values[keep]

    x_mm = x_values * spacing_zyx[2]
    z_mm = first_z_values * spacing_zyx[0]
    slope_z_per_x, intercept = np.polyfit(x_mm, z_mm, deg=1)

    x_mid = (x_size - 1) / 2.0
    left_z = first_z_values[x_values < x_mid]
    right_z = first_z_values[x_values >= x_mid]
    left_first_slice = float(np.median(left_z)) if len(left_z) else None
    right_first_slice = float(np.median(right_z)) if len(right_z) else None

    angle_deg = float(np.degrees(np.arctan(slope_z_per_x)))
    return {
        "first_side": first_side,
        "sample_count": int(len(x_values)),
        "slope_z_per_x_mm": float(slope_z_per_x),
        "intercept_z_mm": float(intercept),
        "tilt_angle_deg": angle_deg,
        "left_first_slice_median": left_first_slice,
        "right_first_slice_median": right_first_slice,
        "left_right_first_slice_diff": None
        if left_first_slice is None or right_first_slice is None
        else float(right_first_slice - left_first_slice),
        "min_voxels_per_x": int(min_voxels_per_x),
        "trim_percentile": float(trim_percentile),
    }


def rotate_volume_to_left_right_first_slice_parallel(
    volume: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    first_side="low",
    min_voxels_per_x=1,
    trim_percentile=10.0,
    max_abs_angle_deg=30.0,
    pad=0,
):
    """
    スライスをz方向に進めたとき、左右が同じタイミングで表示されるように補正する。
    咬合面合わせ後のボリュームに対して使う想定。
    戻り値:
      rotated_uint8, info(dict)
    """
    if volume.ndim != 3:
        raise ValueError(f"3次元配列を想定していますが shape={volume.shape}")

    mask = volume > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("前景が空です。2値化結果を確認してください。")

    if pad > 0:
        mask = np.pad(mask, ((pad, pad), (pad, pad), (pad, pad)), mode="constant")

    before_info = estimate_left_right_first_slice_tilt(
        mask,
        spacing_zyx=spacing_zyx,
        first_side=first_side,
        min_voxels_per_x=min_voxels_per_x,
        trim_percentile=trim_percentile,
    )
    angle_deg = before_info["tilt_angle_deg"]
    if abs(angle_deg) > max_abs_angle_deg:
        raise ValueError(
            f"推定された左右傾きが大きすぎます: {angle_deg:.3f} deg。"
            f" max_abs_angle_deg={max_abs_angle_deg} を確認してください。"
        )

    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    shape = np.array(mask.shape, dtype=np.float64)

    # 物理座標(z,y,x)で、初出ライン [dz, 0, dx] を x軸方向 [0,0,1] へ倒す。
    direction = np.array([before_info["slope_z_per_x_mm"], 0.0, 1.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r_phys = rodrigues_rotation_matrix(direction, target)

    s = np.diag(spacing_zyx)
    s_inv = np.diag(1.0 / spacing_zyx)
    a = s_inv @ r_phys @ s
    m = np.linalg.inv(a)
    center = (shape - 1.0) / 2.0
    offset = center - m @ center

    rotated = ndi.affine_transform(
        mask.astype(np.uint8),
        matrix=m,
        offset=offset,
        output_shape=tuple(mask.shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    rotated = (rotated > 0).astype(np.uint8) * 255

    try:
        after_info = estimate_left_right_first_slice_tilt(
            rotated,
            spacing_zyx=spacing_zyx,
            first_side=first_side,
            min_voxels_per_x=min_voxels_per_x,
            trim_percentile=trim_percentile,
        )
    except ValueError as exc:
        after_info = {"error": str(exc)}

    info = {
        "input_shape_zyx": [int(v) for v in volume.shape],
        "working_shape_zyx": [int(v) for v in mask.shape],
        "spacing_zyx": [float(v) for v in spacing_zyx],
        "first_side": first_side,
        "pad": int(pad),
        "max_abs_angle_deg": float(max_abs_angle_deg),
        "before": before_info,
        "after": after_info,
    }
    return rotated, info


def rotate_volume_to_occlusal_and_left_right_parallel(
    volume: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    percentile=85.0,
    side="auto",
    pad=20,
    inplane_k45=0,
    first_side="low",
    min_voxels_per_x=1,
    trim_percentile=10.0,
    max_abs_left_right_angle_deg=30.0,
):
    """
    既存の咬合平面合わせを行った後、左右の初出スライス差も補正する。
    rotate_volume_to_occlusal_parallel は変更せずに使うため、既存呼び出しには影響しない。
    """
    occlusal_rotated, occlusal_info = rotate_volume_to_occlusal_parallel(
        volume=volume,
        spacing_zyx=spacing_zyx,
        percentile=percentile,
        side=side,
        pad=pad,
        inplane_k45=inplane_k45,
    )
    left_right_rotated, left_right_info = rotate_volume_to_left_right_first_slice_parallel(
        occlusal_rotated,
        spacing_zyx=spacing_zyx,
        first_side=first_side,
        min_voxels_per_x=min_voxels_per_x,
        trim_percentile=trim_percentile,
        max_abs_angle_deg=max_abs_left_right_angle_deg,
        pad=0,
    )
    info = {
        "occlusal": occlusal_info,
        "left_right": left_right_info,
        "workflow": "occlusal plane alignment -> left/right first-slice alignment",
    }
    return left_right_rotated, info


def create_radial_projection_images(
    volume: np.ndarray,
    num_angles: int = 360,
    center: tuple[float, float] | None = None,
    radius: int | None = None,
    interpolation_order: int = 1,
) -> np.ndarray:
    """
    3D volume[z, y, x] から、axial面ごとに中心から画像端までの線分を取得し、
    angleごとに (slice, radius) の2D画像を作成する。
    """
    if volume.ndim != 3:
        raise ValueError("volume must be 3D array with shape (Z, Y, X)")

    z_size, y_size, x_size = volume.shape

    if center is None:
        center_y = (y_size - 1) / 2.0
        center_x = (x_size - 1) / 2.0
    else:
        center_y, center_x = center

    if radius is None:
        radius = int(
            min(
                center_x,
                center_y,
                x_size - 1 - center_x,
                y_size - 1 - center_y,
            )
        )

    radial_positions = np.arange(radius, dtype=np.float32)
    z_indices = np.arange(z_size, dtype=np.float32)
    projections = np.zeros((num_angles, z_size, radius), dtype=np.float32)

    for angle_index in range(num_angles):
        theta = 2.0 * np.pi * angle_index / num_angles
        dx = np.cos(theta)
        dy = np.sin(theta)
        x_line = center_x + radial_positions * dx
        y_line = center_y + radial_positions * dy
        zz, _ = np.meshgrid(z_indices, radial_positions, indexing="ij")
        coords_z = zz
        coords_y = np.broadcast_to(y_line, (z_size, radius))
        coords_x = np.broadcast_to(x_line, (z_size, radius))

        projections[angle_index] = map_coordinates(
            volume,
            [coords_z, coords_y, coords_x],
            order=interpolation_order,
            mode="constant",
            cval=0,
        )

    return projections


def pca_axes(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return center, eigenvectors[:, order[0]], eigenvectors[:, order[1]]


def line_points_from_split_axis(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    split_axis_xy: np.ndarray,
    line_axis_xy: np.ndarray,
) -> np.ndarray:
    positions = (points_xy - center_xy) @ line_axis_xy
    line_t = np.linspace(float(positions.min()), float(positions.max()), 600)
    return center_xy + np.outer(line_t, line_axis_xy)


def split_2d_mask_by_pca(
    mask: np.ndarray,
    *,
    target_axis: str,
    negative_label: str,
    positive_label: str,
) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError("Too few foreground pixels for PCA split.")

    points_xy = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center, pc1, pc2 = pca_axes(points_xy)
    axes = [pc1, pc2]
    component_index = 0 if target_axis == "x" else 1
    split_axis_index = int(np.argmax([abs(axis[component_index]) for axis in axes]))
    split_axis = axes[split_axis_index]
    line_axis = axes[1 - split_axis_index]

    height, width = mask.shape
    yy, xx = np.indices((height, width))
    grid_points = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    signed = ((grid_points - center) @ split_axis).reshape(height, width)

    negative_mask = mask & (signed <= 0)
    positive_mask = mask & (signed > 0)

    if target_axis == "y":
        negative_mean = float(np.mean(np.nonzero(negative_mask)[0])) if np.any(negative_mask) else np.inf
        positive_mean = float(np.mean(np.nonzero(positive_mask)[0])) if np.any(positive_mask) else np.inf
    else:
        negative_mean = float(np.mean(np.nonzero(negative_mask)[1])) if np.any(negative_mask) else np.inf
        positive_mean = float(np.mean(np.nonzero(positive_mask)[1])) if np.any(positive_mask) else np.inf

    if negative_mean <= positive_mean:
        masks = {negative_label: negative_mask, positive_label: positive_mask}
    else:
        masks = {negative_label: positive_mask, positive_label: negative_mask}

    return {
        "center_xy": center,
        "pc1_xy": pc1,
        "pc2_xy": pc2,
        "split_axis_xy": split_axis,
        "line_axis_xy": line_axis,
        "line_points_xy": line_points_from_split_axis(points_xy, center, split_axis, line_axis),
        "masks": masks,
    }


def split_radial_frames_upper_lower(projections: np.ndarray) -> np.ndarray:
    labels = np.zeros(projections.shape, dtype=np.uint8)
    for angle_index in range(projections.shape[0]):
        frame_mask = projections[angle_index] > 0
        if np.count_nonzero(frame_mask) < 20:
            continue
        split = split_2d_mask_by_pca(
            frame_mask,
            target_axis="y",
            negative_label="upper",
            positive_label="lower",
        )
        labels[angle_index][split["masks"]["upper"]] = 1
        labels[angle_index][split["masks"]["lower"]] = 2
    return labels


def restore_radial_labels_to_axial_volume(
    radial_labels: np.ndarray,
    volume_shape: tuple[int, int, int],
    center: tuple[float, float] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    z_size, y_size, x_size = volume_shape
    num_angles, label_z_size, label_radius = radial_labels.shape
    if label_z_size != z_size:
        raise ValueError(f"Z size mismatch: labels={label_z_size}, volume={z_size}")

    if center is None:
        center_y = (y_size - 1) / 2.0
        center_x = (x_size - 1) / 2.0
    else:
        center_y, center_x = center

    if radius is None:
        radius = label_radius

    yy, xx = np.indices((y_size, x_size))
    dx = xx.astype(np.float64) - center_x
    dy = yy.astype(np.float64) - center_y
    rr = np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.int32)
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    aa = np.rint(theta * num_angles / (2.0 * np.pi)).astype(np.int32) % num_angles
    valid = rr < min(radius, label_radius)

    restored = np.zeros(volume_shape, dtype=np.uint8)
    for z_index in range(z_size):
        restored_slice = restored[z_index]
        restored_slice[valid] = radial_labels[aa[valid], z_index, rr[valid]]
    return restored


def width_profile(
    points_xy: np.ndarray,
    center: np.ndarray,
    pc1: np.ndarray,
    pc2: np.ndarray,
    bin_count: int = 80,
    min_points_per_bin: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points_xy - center
    u = centered @ pc1
    v = centered @ pc2
    edges = np.linspace(v.min(), v.max(), bin_count + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    widths = np.full(bin_count, np.nan, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int32)

    for index in range(bin_count):
        in_bin = (v >= edges[index]) & (v < edges[index + 1])
        if index == bin_count - 1:
            in_bin = (v >= edges[index]) & (v <= edges[index + 1])
        counts[index] = int(np.count_nonzero(in_bin))
        if counts[index] < min_points_per_bin:
            continue
        u_in_bin = u[in_bin]
        widths[index] = float(u_in_bin.max() - u_in_bin.min())

    valid = np.isfinite(widths)
    if np.count_nonzero(valid) < 4:
        raise ValueError("Too few valid bins for anterior/molar split.")
    fitted_widths = np.interp(centers, centers[valid], widths[valid])
    return centers, fitted_widths, counts


def orient_anterior_to_low_v(points_xy: np.ndarray, center: np.ndarray, pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    centers, widths, _ = width_profile(points_xy, center, pc1, pc2)
    edge_count = max(3, len(centers) // 5)
    if float(np.nanmedian(widths[:edge_count])) <= float(np.nanmedian(widths[-edge_count:])):
        return pc2
    return -pc2


def estimate_split_v(points_xy: np.ndarray, center: np.ndarray, pc1: np.ndarray, pc2: np.ndarray) -> tuple[float, dict]:
    centers, widths, counts = width_profile(points_xy, center, pc1, pc2)
    bin_count = len(centers)
    edge_count = max(3, bin_count // 5)
    anterior_width = float(np.nanmedian(widths[:edge_count]))
    posterior_width = float(np.nanpercentile(widths[-2 * edge_count :], 75))
    width_gain = posterior_width - anterior_width
    width_target = anterior_width + 0.45 * width_gain
    search_start = max(1, bin_count // 10)
    search_end = max(search_start + 1, int(bin_count * 0.9))
    candidates = np.where(widths[search_start:search_end] >= width_target)[0]
    if len(candidates) > 0 and width_gain > 0:
        split_index = int(candidates[0] + search_start)
        method = "width_threshold"
    else:
        gradient = np.gradient(widths)
        split_index = int(np.argmax(np.abs(gradient[search_start:search_end])) + search_start)
        method = "max_abs_gradient_fallback"
    return float(centers[split_index]), {"method": method, "width_gain": float(width_gain)}


def make_anterior_molar_separator(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError("Too few foreground pixels for anterior/molar split.")
    points_xy = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center, pc1, pc2 = pca_axes(points_xy)
    pc2 = orient_anterior_to_low_v(points_xy, center, pc1, pc2)
    centered = points_xy - center
    side_u = centered @ pc1
    side_v = centered @ pc2
    split_v, split_info = estimate_split_v(points_xy, center, pc1, pc2)

    slope, intercept = np.polyfit(side_v, side_u, deg=1)
    split_u = float(slope * split_v + intercept)
    tangent_uv = np.array([slope, 1.0], dtype=np.float64)
    tangent_uv /= np.linalg.norm(tangent_uv) + 1e-12
    separator_direction_uv = np.array([1.0, -slope], dtype=np.float64)
    separator_direction_uv /= np.linalg.norm(separator_direction_uv) + 1e-12

    line_length = max(20.0, float(np.percentile(np.abs(side_u - split_u), 95)) * 2.4)
    line_t = np.linspace(-line_length, line_length, 600)
    line_u = split_u + line_t * separator_direction_uv[0]
    line_v = split_v + line_t * separator_direction_uv[1]
    line_points_xy = center + np.outer(line_u, pc1) + np.outer(line_v, pc2)
    return {
        "center_xy": center,
        "pc1_xy": pc1,
        "pc2_xy": pc2,
        "split_u": split_u,
        "split_v": float(split_v),
        "tangent_uv": tangent_uv,
        "line_points_xy": line_points_xy,
        "split_info": split_info,
    }


def draw_line_rgb(image_rgb: np.ndarray, line_points_xy: np.ndarray, color: tuple[int, int, int]) -> None:
    height, width = image_rgb.shape[:2]
    rounded = np.rint(line_points_xy).astype(np.int32)
    for x, y in rounded:
        if 0 <= x < width and 0 <= y < height:
            y0 = max(0, y - 1)
            y1 = min(height, y + 2)
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            image_rgb[y0:y1, x0:x1] = color


def build_axial_overlay(
    volume: np.ndarray,
    upper_lower_labels: np.ndarray,
    line_points: dict[str, np.ndarray],
) -> np.ndarray:
    base = np.where(volume > 0, 90, 0).astype(np.uint8)
    overlay = np.repeat(base[..., None], 3, axis=-1)
    overlay[upper_lower_labels == 1] = (80, 140, 255)
    overlay[upper_lower_labels == 2] = (80, 220, 120)

    colors = {
        "upper_left_right": (255, 255, 255),
        "lower_left_right": (220, 220, 220),
        "upper_left_anterior_molar": (255, 40, 40),
        "upper_right_anterior_molar": (255, 160, 40),
        "lower_left_anterior_molar": (255, 40, 160),
        "lower_right_anterior_molar": (255, 220, 40),
    }
    for z_index in range(overlay.shape[0]):
        for name, points in line_points.items():
            draw_line_rgb(overlay[z_index], points, colors.get(name, (255, 255, 255)))
    return overlay


def analyze_radial_projection_workflow(
    volume: np.ndarray,
    num_angles: int = 360,
    center: tuple[float, float] | None = None,
    radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    projections = create_radial_projection_images(
        volume,
        num_angles=num_angles,
        center=center,
        radius=radius,
        interpolation_order=0,
    )
    radial_upper_lower = split_radial_frames_upper_lower(projections)
    restored_upper_lower = restore_radial_labels_to_axial_volume(
        radial_upper_lower,
        volume.shape,
        center=center,
        radius=radius,
    )

    upper_axial = np.any(restored_upper_lower == 1, axis=0)
    lower_axial = np.any(restored_upper_lower == 2, axis=0)
    upper_lr = split_2d_mask_by_pca(upper_axial, target_axis="x", negative_label="left", positive_label="right")
    lower_lr = split_2d_mask_by_pca(lower_axial, target_axis="x", negative_label="left", positive_label="right")

    quadrant_masks = {
        "upper_left": upper_lr["masks"]["left"],
        "upper_right": upper_lr["masks"]["right"],
        "lower_left": lower_lr["masks"]["left"],
        "lower_right": lower_lr["masks"]["right"],
    }
    line_points = {
        "upper_left_right": upper_lr["line_points_xy"],
        "lower_left_right": lower_lr["line_points_xy"],
    }
    for quadrant_name, quadrant_mask in quadrant_masks.items():
        line_points[f"{quadrant_name}_anterior_molar"] = make_anterior_molar_separator(quadrant_mask)["line_points_xy"]

    overlay = build_axial_overlay(volume, restored_upper_lower, line_points)
    return radial_upper_lower, overlay


def save_projection_volume(
    projections: np.ndarray,
    output_path: str | Path,
) -> None:
    """projections shape = (angle, Z, R) を1つの3D TIFFとして保存する。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(output_path, projections.astype(np.float32))


def main():
    """
    parser = argparse.ArgumentParser(
        description="歯2値3Dボリュームを回転し、咬合面をスライス面に平行化する。"
    )
    parser.add_argument("--input", required=True, help="入力3D tif")
    parser.add_argument("--output", required=True, help="出力3D tif")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SZ", "SY", "SX"),
        help="ボクセル間隔 (z y x)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=85.0,
        help="咬合面候補抽出に使う端側パーセンタイル (50-99推奨)",
    )
    parser.add_argument(
        "--side",
        choices=["auto", "high", "low"],
        default="auto",
        help="咬合面候補を z高側/低側/自動で選択",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=20,
        help="回転前パディング幅",
    )
    parser.add_argument(
        "--info-json",
        default="",
        help="回転情報を書き出す json パス（省略可）",
    )
    parser.add_argument(
        "--inplane-k45",
        type=int,
        default=0,
        help="スライス面内の追加回転ステップ数（1で45度、2で90度）",
    )
    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    """
    input_path = r"D:/_study/ImageProcessing/study/Watershed/Data/Input/labeled_map_filled/label_map_8_filled.tif"
    output_path = r"D:/_study/ImageProcessing/study/Watershed/Data/Output/test/label_map_8_filled_aligned.tif"
    spacing_zyx = (1, 1, 1)
    percentile = 95
    side = "auto"
    pad = 20
    inplane_k45 = 7
    first_side = "low"
    info_json = r"D:/_study/ImageProcessing/study/Watershed/Data/Output/test/label_map_8_filled_aligned_info.json"

    vol = tiff.imread(input_path)
    rotated, info = rotate_volume_to_occlusal_and_left_right_parallel(
        volume=vol,
        spacing_zyx=spacing_zyx,
        percentile=percentile,
        side=side,
        pad=pad,
        inplane_k45=inplane_k45,
        first_side=first_side,
    )

    tiff.imwrite(output_path, rotated.astype(np.uint8), dtype=np.uint8)

    if info_json:
        p = Path(info_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", str(output_path))
    print("Occlusal rotation angle [deg]:", f"{info['occlusal']['rotation_angle_deg']:.3f}")
    print("In-plane angle [deg]:", f"{info['occlusal']['inplane_angle_deg']:.3f}")
    print("Left-right tilt before [deg]:", f"{info['left_right']['before']['tilt_angle_deg']:.3f}")
    after = info["left_right"].get("after", {})
    if "tilt_angle_deg" in after:
        print("Left-right tilt after [deg]:", f"{after['tilt_angle_deg']:.3f}")


if __name__ == "__main__":
    main()
