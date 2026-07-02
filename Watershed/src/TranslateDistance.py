import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

# 距離画像にするよ
def save_distance_image(
    input_tif_path: str,
    output_tif_path: str,
    normalize_for_view: bool = False,
    mode: str = "to_seed",
    distance_dim: int = 3,
) -> np.ndarray:
    """3Dマスクから距離画像を作成して保存する。

    引数:
        input_tif_path: 入力tifパス。0以外を前景として扱う。
        output_tif_path: 出力tifパス。
        normalize_for_view: Trueなら16bit正規化画像として保存する。
            Falseなら生の距離値をfloat32で保存する。
        mode: 距離の定義。
            "inside": 前景内部から背景までの距離(背景は0)。
            "to_seed": 全ボクセルから前景(seed)までの距離。
        distance_dim: 距離変換の次元数。2または3を指定する。
            2なら3D入力時に各zスライスごとに2D距離変換を行う。
            3ならボリューム全体で3D距離変換を行う。
    返り値:
        生の距離画像(float配列)。
    """
    volume = tiff.imread(input_tif_path)
    bw = volume > 0

    if distance_dim not in (2, 3):
        raise ValueError("distance_dim must be 2 or 3.")

    # 2D入力は常に2D距離変換として扱う。
    if bw.ndim == 2:
        distance_dim = 2

    if mode == "inside":
        mask_for_dt = bw
    elif mode == "to_seed":
        mask_for_dt = ~bw
    else:
        raise ValueError("mode must be 'inside' or 'to_seed'.")

    if distance_dim == 3:
        if mask_for_dt.ndim != 3:
            raise ValueError("distance_dim=3 requires 3D input volume.")
        dist = ndi.distance_transform_edt(mask_for_dt)
    else:
        if mask_for_dt.ndim == 2:
            dist = ndi.distance_transform_edt(mask_for_dt)
        elif mask_for_dt.ndim == 3:
            dist = np.zeros(mask_for_dt.shape, dtype=np.float32)
            for z in range(mask_for_dt.shape[0]):
                dist[z] = ndi.distance_transform_edt(mask_for_dt[z]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported input ndim: {mask_for_dt.ndim}")

    if normalize_for_view:
        dist_to_save = (dist / (dist.max() + 1e-8) * 65535).astype(np.uint16)
    else:
        dist_to_save = dist.astype(np.float32)

    tiff.imwrite(output_tif_path, dist_to_save)
    return dist

if __name__ == "__main__":
    input_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\combined_anterior_molar_seed_volume_distance.tif"
    distance = save_distance_image(
        input_tif_path=input_path,
        output_tif_path=output_path,
        normalize_for_view=True,
        mode="inside",
        distance_dim=2
    )