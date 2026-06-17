import numpy as np
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

import numpy as np
import tifffile as tiff

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
    import tifffile as tiff

    path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\label_map_8_filled_aligned.tif"
    vol = tiff.imread(path)

    # 各スライス1点の重心
    centroids = detect_slice_centroids(vol, per_component=False)
    save_centroids_overlay_tiff(
        vol,
        centroids,
        r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\centroids_overlay_single.tif",
        radius=2
    )

    # 連結成分ごとの重心
    cc_centroids = detect_slice_centroids(vol, per_component=True, connectivity=2)
    save_centroids_overlay_tiff(
        vol,
        cc_centroids,
        r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\centroids_overlay_components.tif",
        radius=2
    )