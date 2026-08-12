# 根管を領域拡張を用いて抽出する
import numpy as np
from scipy import ndimage as ndi

def labeled_for_root_canal(volume: np.ndarray, target_value: int = 1) -> np.ndarray:
    """
    こんかんのラベル付け

    Parameters:
        volume (np.ndarray): 入力の3D画像データ
        target_value (int) : 抽出対象の値（現状 1）

    Returns:
        np.ndarray: ラベル付きの3D画像データ

    Details:
        - volume 内の target_value の領域を抽出しラベリングをする
        - 周囲 5 ボクセルに対象があれば、同一のラベルとする
    """

    structure_nbh = structure_element(size=5)
    labeled, label_numbers = ndi.label(volume == target_value, structure=structure_nbh)

    return labeled

def structure_element(size: int = 2) -> np.ndarray:
    """
    立方体の構造要素を生成する関数

    Parameters:
        size (int): 近傍のサイズ（デフォルトは 2 ボクセル）

    Returns:
        np.ndarray: 立方体の構造要素
    """
    radius = size
    size = 2 * radius + 1
    center = radius
    structure = np.zeros((size, size, size), dtype=bool)

    for z in range(size):
        for y in range(size):
            for x in range(size):
                if max(abs(z - center), abs(y - center), abs(x - center)) <= radius:
                    structure[z, y, x] = True
    return structure
