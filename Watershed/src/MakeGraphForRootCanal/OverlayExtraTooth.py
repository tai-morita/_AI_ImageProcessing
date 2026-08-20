# 元の Volume データに対して、歯のデータ(2 値化)だけを抽出する

import numpy as np
from .DataLoad import load_data, save_volume
def overlay_extra_tooth(volume: np.ndarray, extra_tooth: np.ndarray) -> np.ndarray:
    """
    元の Volume データに対して、歯のデータ(2 値化)だけを抽出する関数

    Parameters:
        volume (np.ndarray)     : 入力の3D画像データ
        extra_tooth (np.ndarray): 歯の抽出データ(2 値化)

    Returns:
        np.ndarray: 元の Volume データに対して、歯のデータ(2 値化)だけを抽出した 3D画像データ
    """
    if volume.shape != extra_tooth.shape:
        raise ValueError(f"volume と extra_tooth の形状が一致しません: volume.shape={volume.shape}, extra_tooth.shape={extra_tooth.shape}")

    extracted_volume = np.where(extra_tooth == 1, volume, 0)

    return extracted_volume