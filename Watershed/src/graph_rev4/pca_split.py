import numpy as np


def pca_axes_xy(points_xy: np.ndarray):
    """2次元点群からPCA主軸を計算する。

    Args:
        points_xy: (N, 2) 形式のXY座標点群。

    Returns:
        重心、第一主成分、第二主成分、固有値降順配列。
    """
    if len(points_xy) < 3:
        raise ValueError("PCA requires at least 3 points.")
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return center, eigenvectors[:, order[0]], eigenvectors[:, order[1]], eigenvalues[order]


def axial_projection_points_xy(volume: np.ndarray):
    """3次元ボリュームのaxial投影から前景点群を抽出する。

    Args:
        volume: 入力3次元ボリューム。

    Returns:
        2次元投影マスクと投影面上の前景点群XY座標。
    """
    mask = volume > 0
    projection = np.any(mask, axis=0)
    ys, xs = np.nonzero(projection)
    if len(xs) < 20:
        raise ValueError("Too few foreground points in axial projection.")
    points_xy = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    return projection, points_xy


def split_volume_anterior_molar_by_pca(
    volume: np.ndarray,
    split_percentile: float = 50.0,
    ambiguous_band_px: float = 0.0,
):
    """PCA軸を使って前歯領域と臼歯領域に分割する。

    Args:
        volume: 入力3次元ボリューム。
        split_percentile: 分割境界スコアに用いる百分位。
        ambiguous_band_px: 境界曖昧帯の幅。

    Returns:
        前歯マスク、臼歯マスク、分割ラベルボリューム、分割ライン座標、分割情報辞書。
    """
    projection, points_xy = axial_projection_points_xy(volume)
    center_xy, pc1_xy, pc2_xy, eigenvalues = pca_axes_xy(points_xy)
    axes = [pc1_xy, pc2_xy]
    split_axis_index = int(np.argmax([abs(axis[1]) for axis in axes]))
    split_axis_xy = axes[split_axis_index].copy()
    along_arch_axis_xy = axes[1 - split_axis_index].copy()

    if split_axis_xy[1] < 0:
        split_axis_xy = -split_axis_xy

    scores = (points_xy - center_xy) @ split_axis_xy
    split_score = float(np.percentile(scores, split_percentile))

    height, width = projection.shape
    yy, xx = np.indices((height, width))
    grid_points = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    grid_scores = ((grid_points - center_xy) @ split_axis_xy).reshape(height, width)

    anterior_xy_mask = projection & (grid_scores <= split_score - ambiguous_band_px)
    molar_xy_mask = projection & (grid_scores >= split_score + ambiguous_band_px)
    if ambiguous_band_px <= 0:
        anterior_xy_mask = projection & (grid_scores <= split_score)
        molar_xy_mask = projection & (grid_scores > split_score)

    volume_mask = volume > 0
    anterior_volume = volume_mask & anterior_xy_mask[None, :, :]
    molar_volume = volume_mask & molar_xy_mask[None, :, :]

    split_label_volume = np.zeros_like(volume, dtype=np.uint8)
    split_label_volume[anterior_volume] = 1
    split_label_volume[molar_volume] = 2
    if ambiguous_band_px > 0:
        ambiguous_xy_mask = projection & ~(anterior_xy_mask | molar_xy_mask)
        split_label_volume[volume_mask & ambiguous_xy_mask[None, :, :]] = 3

    line_t = np.linspace(-max(height, width), max(height, width), 600)
    split_line_xy = center_xy + split_score * split_axis_xy + np.outer(line_t, along_arch_axis_xy)

    info = {
        "center_xy": center_xy.astype(float).tolist(),
        "pc1_xy": pc1_xy.astype(float).tolist(),
        "pc2_xy": pc2_xy.astype(float).tolist(),
        "eigenvalues_desc": eigenvalues.astype(float).tolist(),
        "split_axis_index": int(split_axis_index),
        "split_axis_xy": split_axis_xy.astype(float).tolist(),
        "along_arch_axis_xy": along_arch_axis_xy.astype(float).tolist(),
        "split_score": float(split_score),
        "split_percentile": float(split_percentile),
        "ambiguous_band_px": float(ambiguous_band_px),
        "label_definition": {
            "0": "background",
            "1": "anterior/image-upper",
            "2": "molar/image-lower",
            "3": "ambiguous",
        },
        "anterior_voxel_count": int(np.count_nonzero(anterior_volume)),
        "molar_voxel_count": int(np.count_nonzero(molar_volume)),
    }
    return anterior_volume, molar_volume, split_label_volume, split_line_xy, info


def make_anterior_molar_split_overlay_volume(
    volume: np.ndarray,
    split_line_xy: np.ndarray,
    line_radius_px: int = 2,
):
    """分割ラインを重畳した可視化ボリュームを作成する。

    Args:
        volume: 入力3次元ボリューム。
        split_line_xy: 分割ライン上のXY座標列。
        line_radius_px: 分割ライン描画時の半径ピクセル数。

    Returns:
        RGBのオーバーレイボリューム。
    """
    foreground = volume > 0
    base = np.zeros_like(volume, dtype=np.uint8)
    if np.any(foreground):
        foreground_values = volume[foreground].astype(np.float32)
        scale_max = float(np.percentile(foreground_values, 99))
        if scale_max <= 0:
            scale_max = float(np.max(foreground_values))
        base[foreground] = np.clip(
            volume[foreground].astype(np.float32) / scale_max * 180,
            20,
            180,
        ).astype(np.uint8)

    overlay = np.repeat(base[..., None], 3, axis=-1)

    height, width = volume.shape[1], volume.shape[2]
    line_points = np.rint(split_line_xy).astype(np.int32)
    for x, y in line_points:
        if not (0 <= x < width and 0 <= y < height):
            continue
        y0 = max(0, y - line_radius_px)
        y1 = min(height, y + line_radius_px + 1)
        x0 = max(0, x - line_radius_px)
        x1 = min(width, x + line_radius_px + 1)
        overlay[:, y0:y1, x0:x1] = (255, 40, 40)

    overlay[volume <= 0] = 0
    return overlay


def summarize_region_boundary_slices_from_split_labels(split_label_volume: np.ndarray):
    """分割ラベルごとの存在スライスと連続範囲を要約する。

    Args:
        split_label_volume: 0=背景, 1=前歯, 2=臼歯, 3=曖昧帯のラベルボリューム。

    Returns:
        ラベル別の存在スライス一覧と連続範囲を持つ辞書。
    """
    def _slice_indices_for_label(label_value: int):
        """指定ラベルが存在する1-basedスライス番号一覧を返す。

        Args:
            label_value: 抽出対象ラベル値。

        Returns:
            1-basedスライス番号のリスト。
        """
        present = np.any(split_label_volume == int(label_value), axis=(1, 2))
        return [int(index + 1) for index in np.where(present)[0]]

    def _to_ranges(indices_1based: list[int]):
        """1-basedスライス番号列を連続区間に圧縮する。

        Args:
            indices_1based: 1-basedスライス番号の昇順リスト。

        Returns:
            (start, end) 形式の連続範囲リスト。
        """
        if len(indices_1based) == 0:
            return []
        ranges = []
        start = int(indices_1based[0])
        prev = int(indices_1based[0])
        for current in indices_1based[1:]:
            current = int(current)
            if current == prev + 1:
                prev = current
                continue
            ranges.append((start, prev))
            start = current
            prev = current
        ranges.append((start, prev))
        return ranges

    anterior_slices = _slice_indices_for_label(1)
    molar_slices = _slice_indices_for_label(2)
    boundary_slices = _slice_indices_for_label(3)

    return {
        "anterior_slices_1based": anterior_slices,
        "molar_slices_1based": molar_slices,
        "boundary_slices_1based": boundary_slices,
        "anterior_slice_ranges_1based": _to_ranges(anterior_slices),
        "molar_slice_ranges_1based": _to_ranges(molar_slices),
        "boundary_slice_ranges_1based": _to_ranges(boundary_slices),
    }
