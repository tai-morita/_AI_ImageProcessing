# 根管を二値化で抽出する
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi


def extract_root_canal(volume: np.ndarray, threshold: int = 1000) -> np.ndarray:
    """
    根管を二値化で抽出する関数

    Parameters:
        volume (np.ndarray): 入力の3D画像データ
        threshold (int)  : 二値化の閾値（現状 1000）

    Returns:
        np.ndarray: 0: 背景, 1: 根管, 2: その他の領域
    """
    volume = np.where(
        volume == 0,
        0,
        np.where(volume > threshold, 2, 1)
    ).astype(np.uint16)

    # 膨張してノイズ除去
    mask = (volume == 1)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    volume[mask] = 1

    return volume