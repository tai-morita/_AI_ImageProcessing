import numpy as np
from scipy import ndimage as ndi

def labeled_for_root_canal(volume: np.ndarray, target_value: int = 1, dilation_iterations: int = 5) -> np.ndarray:
    """
    根管のラベル付け

    Parameters:
        volume (np.ndarray): 入力の3D画像データ
        target_value (int) : 抽出対象の値（現状 1）
        dilation_iterations (int): 膨張の反復回数（デフォルトは 5）

    Returns:
        np.ndarray: ラベル付きの3D画像データ

    Details:
        - volume 内の target_value の領域を抽出しラベリングをする
        - 周囲 5 ボクセルに対象があれば、同一のラベルとする
    """

    return label_nearby_regions(
        volume == target_value,
        dilation_iterations=dilation_iterations,
    )


def label_nearby_regions(
    mask: np.ndarray,
    dilation_iterations: int = 5, # 5 ボクセルまで膨張させてラベル付けする
) -> np.ndarray:
    """Expand nearby 3D regions, then label only the original foreground voxels."""
    # 隣接領域を膨張させてラベル付けする
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape}")
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    connectivity = ndi.generate_binary_structure(rank=3, connectivity=3)
    expanded_mask = ndi.binary_dilation(
        mask,
        structure=connectivity,
        iterations=dilation_iterations,
    )
    expanded_labels, _ = ndi.label(expanded_mask, structure=connectivity)
    return np.where(mask, expanded_labels, 0)
