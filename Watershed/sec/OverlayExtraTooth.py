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

def ternary_image_0B1R2T(extracted_volume: np.ndarray, root_canal_binary: np.ndarray) -> np.ndarray:
    """
    元の Volume データに対して、歯のデータ(2 値化)だけを抽出し、根管の二値化画像を重ね合わせる関数

    Parameters:
        extracted_volume (np.ndarray): 元の Volume データに対して、歯のデータ(2 値化)だけを抽出した 3D画像データ
        root_canal_binary (np.ndarray): 根管の二値化画像

    Returns:
        np.ndarray: 歯が 2, 根管が 1, 背景が 0 の 3 値画像
    """
    if extracted_volume.shape != root_canal_binary.shape:
        raise ValueError(f"extracted_volume と root_canal_binary の形状が一致しません: extracted_volume.shape={extracted_volume.shape}, root_canal_binary.shape={root_canal_binary.shape}")

    ternary_image = np.where(root_canal_binary == 1, 1, np.where(extracted_volume > 0, 2, 0))

    return ternary_image