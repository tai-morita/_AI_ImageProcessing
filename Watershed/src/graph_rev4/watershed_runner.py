import os

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import feature, filters, segmentation, util

try:
    from .visualize import save_colorized_labels_local
except ImportError:
    from visualize import save_colorized_labels_local


def _distance_transform_edt_in_bbox(mask_3d: np.ndarray) -> np.ndarray:
    """前景領域のバウンディングボックス内だけでEDTを計算する。

    Args:
        mask_3d: 3次元の2値マスク。

    Returns:
        mask_3dと同shapeの距離画像。前景外は0。
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask but got {mask_3d.shape}.")

    if not np.any(mask_3d):
        return np.zeros(mask_3d.shape, dtype=np.float32)

    zyx = np.argwhere(mask_3d)
    z_min, y_min, x_min = zyx.min(axis=0)
    z_max, y_max, x_max = zyx.max(axis=0)

    cropped = mask_3d[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    cropped_dist = ndi.distance_transform_edt(cropped)

    dist = np.zeros(mask_3d.shape, dtype=np.float32)
    dist[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = cropped_dist.astype(
        np.float32
    )
    return dist


def watershed_3d_volume(
    volume: np.ndarray,
    seed_volume: np.ndarray,
    labels_out: str,
    connectivity: int = 6,
    use_bbox_acceleration: bool = True,
    save_distance_volume: bool = True,
) -> np.ndarray:
    """3次元watershedを実行してラベル画像を保存する。

    Args:
        volume: 入力3次元ボリューム。
        seed_volume: マーカーseedボリューム。Noneの場合は自動生成。
        labels_out: ラベル画像の出力パス。
        connectivity: 3次元連結性。
        use_bbox_acceleration: Trueなら前景bbox内でEDTを計算して高速化する。
        save_distance_volume: Trueなら距離画像を保存する。

    Returns:
        生成したラベルボリューム。
    """
    vol = volume
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume but got {vol.shape}.")

    output_dir = os.path.dirname(labels_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    vol = util.img_as_float(vol)
    _ = filters.gaussian(vol, sigma=1.0, preserve_range=True)

    bw = vol > 0
    if use_bbox_acceleration:
        dist = _distance_transform_edt_in_bbox(bw)
    else:
        dist = ndi.distance_transform_edt(bw).astype(np.float32)
    if save_distance_volume:
        dist_out = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(labels_out))[0]}_dist.tif",
        )
        tiff.imwrite(dist_out, dist.astype(np.float32))

    if seed_volume is None:
        coords = feature.peak_local_max(
            dist,
            labels=bw,
            footprint=np.ones((3, 3, 3), dtype=bool),
            exclude_border=False,
        )
        markers = np.zeros_like(dist, dtype=np.int32)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)
    else:
        markers = seed_volume.astype(np.int32)

    labels = segmentation.watershed(
        -dist,
        markers=markers,
        mask=bw,
        connectivity=connectivity,
        watershed_line=False,
    )

    tiff.imwrite(labels_out, labels.astype(np.uint16))
    base, ext = os.path.splitext(labels_out)
    color_out = base + "_color" + ext
    save_colorized_labels_local(labels, color_out)

    return labels
