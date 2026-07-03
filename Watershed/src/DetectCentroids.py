import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

def detect_slice_centroids(volume: np.ndarray, per_component: bool = False, connectivity: int = 2):
    """
    volume: (Z, Y, X) の3D配列。前景は >0 とみなす。
    per_component=False:
        各スライスで前景全体の重心を1つ返す
    per_component=True:
        各スライスで連結成分ごとの重心を返す
    戻り値:
        per_component=False -> list[dict]
        per_component=True  -> list[list[dict]]
    """
    if volume.ndim != 3:
        raise ValueError(f"3D配列を想定していますが shape={volume.shape}")

    binary = volume > 0
    z_size = binary.shape[0]

    if not per_component:
        results = []
        for z in range(z_size):
            sl = binary[z]
            if not np.any(sl):
                results.append({
                    "slice_index_1based": z + 1,
                    "centroid_y": np.nan,
                    "centroid_x": np.nan,
                    "area": 0
                })
                continue

            ys, xs = np.nonzero(sl)
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
            results.append({
                "slice_index_1based": z + 1,
                "centroid_y": cy,
                "centroid_x": cx,
                "area": int(sl.sum())
            })
        return results

    # 連結成分ごとの重心
    structure = ndi.generate_binary_structure(rank=2, connectivity=connectivity)
    all_slices = []
    for z in range(z_size):
        sl = binary[z]
        labeled, n = ndi.label(sl, structure=structure)
        one_slice = []
        if n == 0:
            all_slices.append(one_slice)
            continue

        centers = ndi.center_of_mass(sl, labeled, range(1, n + 1))
        for label_id, (cy, cx) in enumerate(centers, start=1):
            area = int(np.sum(labeled == label_id))
            one_slice.append({
                "slice_index_1based": z + 1,
                "label_id": label_id,
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "area": area
            })
        all_slices.append(one_slice)

    return all_slices


def extract_centroid_points_3d(
    volume: np.ndarray,
    per_component: bool = True,
    connectivity: int = 2,
) -> np.ndarray:
    """3D画像をスライス方向に潰して、重心点列を (z, y, x) で返す。"""
    centroids = detect_slice_centroids(volume, per_component=per_component, connectivity=connectivity)

    points: list[list[float]] = []
    if per_component:
        for one_slice in centroids:
            for row in one_slice:
                cy = row["centroid_y"]
                cx = row["centroid_x"]
                if np.isnan(cy) or np.isnan(cx):
                    continue
                points.append([
                    float(row["slice_index_1based"] - 1),
                    float(cy),
                    float(cx),
                ])
    else:
        for row in centroids:
            cy = row["centroid_y"]
            cx = row["centroid_x"]
            if np.isnan(cy) or np.isnan(cx):
                continue
            points.append([
                float(row["slice_index_1based"] - 1),
                float(cy),
                float(cx),
            ])

    if not points:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def fit_centroids_quartic(points_3d: np.ndarray):
    """重心点列を z を説明変数として4次近似する。"""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3) as [z, y, x].")
    if points_3d.shape[0] < 5:
        raise ValueError("4次近似には少なくとも5点必要です。")

    z = points_3d[:, 0]
    y = points_3d[:, 1]
    x = points_3d[:, 2]

    coef_y = np.polyfit(z, y, deg=4)
    coef_x = np.polyfit(z, x, deg=4)

    y_fit = np.polyval(coef_y, z)
    x_fit = np.polyval(coef_x, z)
    fitted_points = np.column_stack([z, y_fit, x_fit])

    return {
        "coef_y": coef_y,
        "coef_x": coef_x,
        "fitted_points": fitted_points,
    }


def detect_and_fit_centroids_quartic(
    volume: np.ndarray,
    per_component: bool = True,
    connectivity: int = 2,
):
    """重心点抽出と4次近似をまとめて実行する。"""
    points_3d = extract_centroid_points_3d(
        volume,
        per_component=per_component,
        connectivity=connectivity,
    )
    fit_result = fit_centroids_quartic(points_3d)
    fit_result["points_3d"] = points_3d
    return fit_result


def fit_projected_centroids_quartic_2d(points_2d: np.ndarray):
    """2D重心点列 (y, x) に対して y=f(x) の4次近似を行う。"""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (N, 2) as [y, x].")
    if points_2d.shape[0] < 5:
        raise ValueError("4次近似には少なくとも5点必要です。")

    y = points_2d[:, 0]
    x = points_2d[:, 1]
    unique_x = np.unique(x)
    if unique_x.size < 5:
        raise ValueError("x座標の一意値が少ないため y=f(x) の4次近似が不安定です。")

    coef_y_from_x = np.polyfit(x, y, deg=4)
    y_fit = np.polyval(coef_y_from_x, x)
    fitted_points = np.column_stack([y_fit, x])
    return {
        "coef_y_from_x": coef_y_from_x,
        "fitted_points_2d": fitted_points,
    }


def _select_one_point_per_x_bin(
    points_2d: np.ndarray,
    target_jaw: str = "lower",
    x_bin_width: float = 1.0,
) -> np.ndarray:
    """xビンごとに1点へ集約し、lowerなら最大y、upperなら最小yを採用する。"""
    if target_jaw not in {"lower", "upper"}:
        raise ValueError(f"target_jaw must be 'lower' or 'upper': {target_jaw}")
    if x_bin_width <= 0:
        raise ValueError("x_bin_width must be positive.")

    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (N, 2) as [y, x].")
    if points_2d.shape[0] == 0:
        return points_2d

    y = points_2d[:, 0]
    x = points_2d[:, 1]
    x_bins = np.floor(x / x_bin_width).astype(np.int64)

    selected_idx = []
    for b in np.unique(x_bins):
        idx = np.where(x_bins == b)[0]
        if idx.size == 0:
            continue
        y_vals = y[idx]
        pick_local = int(np.argmax(y_vals)) if target_jaw == "lower" else int(np.argmin(y_vals))
        selected_idx.append(int(idx[pick_local]))

    if not selected_idx:
        return np.empty((0, 2), dtype=np.float64)

    selected = points_2d[np.asarray(selected_idx, dtype=np.int64)]
    order = np.argsort(selected[:, 1])
    return selected[order]


def fit_projected_centroids_quartic_2d_robust(
    points_2d: np.ndarray,
    outlier_sigma: float = 3.0,
    max_iter: int = 5,
):
    """2D重心点列に対して4次近似を行い、MADベースで外れ値を反復除外する。"""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (N, 2) as [y, x].")
    if points_2d.shape[0] < 5:
        raise ValueError("4次近似には少なくとも5点必要です。")

    all_idx = np.arange(points_2d.shape[0], dtype=np.int64)
    active_idx = all_idx.copy()
    active = points_2d.copy()

    for _ in range(max_iter):
        fit_tmp = fit_projected_centroids_quartic_2d(active)
        coef = fit_tmp["coef_y_from_x"]
        x = active[:, 1]
        y = active[:, 0]
        resid = y - np.polyval(coef, x)

        resid_med = np.median(resid)
        abs_dev = np.abs(resid - resid_med)
        mad = np.median(abs_dev)
        if mad <= 1e-12:
            break

        robust_sigma = 1.4826 * mad
        keep = abs_dev <= (outlier_sigma * robust_sigma)

        if np.all(keep):
            break
        if np.count_nonzero(keep) < 5:
            break

        active = active[keep]
        active_idx = active_idx[keep]

    fit_final = fit_projected_centroids_quartic_2d(active)
    inlier_mask = np.zeros(points_2d.shape[0], dtype=bool)
    inlier_mask[active_idx] = True
    outlier_mask = ~inlier_mask

    fit_final["inlier_points_2d"] = active
    fit_final["inlier_mask"] = inlier_mask
    fit_final["outlier_mask"] = outlier_mask
    fit_final["inlier_count"] = int(np.count_nonzero(inlier_mask))
    fit_final["outlier_count"] = int(np.count_nonzero(outlier_mask))
    return fit_final


def _draw_disk_2d(rgb: np.ndarray, y: float, x: float, radius: int, color: tuple[int, int, int]):
    if np.isnan(y) or np.isnan(x):
        return

    h, w, _ = rgb.shape
    yi = int(round(y))
    xi = int(round(x))
    if yi < 0 or yi >= h or xi < 0 or xi >= w:
        return

    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - yi) ** 2 + (xx - xi) ** 2 <= radius ** 2

    patch = rgb[y0:y1, x0:x1]
    patch[..., 0][mask] = color[0]
    patch[..., 1][mask] = color[1]
    patch[..., 2][mask] = color[2]


def _draw_line_2d(
    rgb: np.ndarray,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    color: tuple[int, int, int],
    thickness: int = 1,
):
    if any(np.isnan(v) for v in (y0, x0, y1, x1)):
        return

    length = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    if length <= 1:
        _draw_disk_2d(rgb, y0, x0, radius=max(1, thickness), color=color)
        return

    ys = np.linspace(y0, y1, length)
    xs = np.linspace(x0, x1, length)
    for yy, xx in zip(ys, xs):
        _draw_disk_2d(rgb, yy, xx, radius=max(1, thickness), color=color)


def save_flattened_quartic_overlay_tiff(
    volume: np.ndarray,
    output_path: str,
    connectivity: int = 2,
    point_radius: int = 2,
    curve_thickness: int = 1,
    target_jaw: str = "lower",
    x_bin_width: float = 1.0,
    outlier_sigma: float = 3.0,
    outlier_max_iter: int = 5,
):
    """スライス方向投影画像の上に重心点と4次近似曲線を重ねて保存する。

    手順:
    1) volume を z 方向に最大値投影して2D化
    2) 各スライスのConnComp重心点を抽出し2Dに投影
    3) xビンごとに target_jaw 側の点だけ残す（lower: 最大y / upper: 最小y）
    4) 残した点に対して外れ値除去付き y=f(x) 4次近似
      4) 投影画像の上に重心点(赤)と近似曲線(緑)を重ねる
    """
    if volume.ndim != 3:
        raise ValueError(f"3D配列を想定していますが shape={volume.shape}")

    # z方向へ潰した2D画像(最大値投影)
    projection = np.max(volume > 0, axis=0)
    base_2d = np.where(projection, 255, 0).astype(np.uint8)
    rgb = np.repeat(base_2d[..., None], 3, axis=-1)

    fit_result = detect_and_fit_centroids_quartic(
        volume,
        per_component=True,
        connectivity=connectivity,
    )
    points_3d = fit_result["points_3d"]
    points_2d = points_3d[:, 1:3]

    # 上顎混入を抑えるため、xビンごとに顎側を選別
    jaw_points_2d = _select_one_point_per_x_bin(
        points_2d,
        target_jaw=target_jaw,
        x_bin_width=x_bin_width,
    )

    if jaw_points_2d.shape[0] < 5:
        raise ValueError("顎側選別後の点数が不足しています。x_bin_widthを小さくするか入力を確認してください。")

    fit_2d = fit_projected_centroids_quartic_2d_robust(
        jaw_points_2d,
        outlier_sigma=outlier_sigma,
        max_iter=outlier_max_iter,
    )
    coef_y_from_x = fit_2d["coef_y_from_x"]
    inlier_points_2d = fit_2d["inlier_points_2d"]
    outlier_mask = fit_2d["outlier_mask"]

    # 顎側選別点(赤)と除外点(青)
    for y, x in jaw_points_2d:
        _draw_disk_2d(rgb, y, x, radius=point_radius, color=(255, 0, 0))
    for y, x in jaw_points_2d[outlier_mask]:
        _draw_disk_2d(rgb, y, x, radius=point_radius, color=(0, 0, 255))

    # 近似曲線(緑): xを高密度サンプリングして2D上に描画
    x_min = float(np.min(inlier_points_2d[:, 1]))
    x_max = float(np.max(inlier_points_2d[:, 1]))
    num_samples = max(128, int((x_max - x_min + 1) * 4))
    x_dense = np.linspace(x_min, x_max, num_samples)
    y_dense = np.polyval(coef_y_from_x, x_dense)

    for i in range(1, num_samples):
        _draw_line_2d(
            rgb,
            y_dense[i - 1],
            x_dense[i - 1],
            y_dense[i],
            x_dense[i],
            color=(0, 255, 0),
            thickness=curve_thickness,
        )

    tiff.imwrite(output_path, rgb, photometric="rgb")
    return {
        "output_path": output_path,
        "coef_y_from_x": coef_y_from_x,
        "target_jaw": target_jaw,
        "num_points_all": int(points_2d.shape[0]),
        "num_points_after_jaw_filter": int(jaw_points_2d.shape[0]),
        "num_inliers": int(fit_2d["inlier_count"]),
        "num_outliers": int(fit_2d["outlier_count"]),
    }

def save_centroids_overlay_tiff(
    volume: np.ndarray,
    centroids,
    output_path: str,
    radius: int = 2
):
    """
    volume: (Z, Y, X) 2値/ラベル画像
    centroids:
      - per_component=False の結果: list[dict]
      - per_component=True  の結果: list[list[dict]]
    output_path: RGB TIFF 出力先
    radius: 赤点の半径(pixel)
    """
    if volume.ndim != 3:
        raise ValueError(f"3D配列を想定していますが shape={volume.shape}")

    # 背景黒、前景白のベース画像を作って RGB 化
    base = np.where(volume > 0, 255, 0).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=-1)  # (Z, Y, X, 3)
    z_size, h, w = base.shape

    def draw_red_dot(z_idx: int, cy: float, cx: float):
        if np.isnan(cy) or np.isnan(cx):
            return
        if not (0 <= z_idx < z_size):
            return

        y = int(round(cy))
        x = int(round(cx))
        if not (0 <= y < h and 0 <= x < w):
            return

        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2

        patch = rgb[z_idx, y0:y1, x0:x1]
        patch[..., 0][mask] = 255  # R
        patch[..., 1][mask] = 0    # G
        patch[..., 2][mask] = 0    # B

    # centroids の形式を判定して描画
    if len(centroids) == 0:
        tiff.imwrite(output_path, rgb, photometric="rgb")
        return

    # per_component=False: list[dict]
    if isinstance(centroids[0], dict):
        for row in centroids:
            z = int(row["slice_index_1based"]) - 1
            draw_red_dot(z, row["centroid_y"], row["centroid_x"])
    else:
        # per_component=True: list[list[dict]]
        for one_slice in centroids:
            for row in one_slice:
                z = int(row["slice_index_1based"]) - 1
                draw_red_dot(z, row["centroid_y"], row["centroid_x"])

    tiff.imwrite(output_path, rgb, photometric="rgb")

if __name__ == "__main__":
    path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\combined_jaw_seed_after_edit.tif"
    vol = tiff.imread(path)

    # 各スライス1点の重心
    centroids = detect_slice_centroids(vol, per_component=False)
    save_centroids_overlay_tiff(
        vol,
        centroids,
        r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\combined_jaw\centroids_overlay_single.tif",
        radius=2
    )

    # 連結成分ごとの重心
    cc_centroids = detect_slice_centroids(vol, per_component=True, connectivity=2)
    save_centroids_overlay_tiff(
        vol,
        cc_centroids,
        r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\combined_jaw\centroids_overlay_components.tif",
        radius=2
    )

    # 3D重心点列の抽出 + 4次近似
    fit_result = detect_and_fit_centroids_quartic(vol, per_component=True, connectivity=2)
    print("coef_y:", fit_result["coef_y"])
    print("coef_x:", fit_result["coef_x"])

    # スライス方向投影画像上に重心点と4次近似曲線を重ねて保存
    overlay_result = save_flattened_quartic_overlay_tiff(
        vol,
        output_path=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\combined_jaw\centroids_quartic_overlay_2d.tif",
        connectivity=2,
        point_radius=2,
        curve_thickness=1,
    )
    print("quartic overlay output:", overlay_result["output_path"])