import json
import os

import numpy as np
import tifffile as tiff


def ensure_dir(path: str) -> None:
    """ディレクトリが存在しない場合に作成する。

    Args:
        path: 作成対象のディレクトリパス。

    Returns:
        なし。
    """
    os.makedirs(path, exist_ok=True)


def load_volume(input_path: str) -> np.ndarray:
    """3次元ボリュームをTIFFから読み込む。

    Args:
        input_path: 入力TIFFファイルのパス。

    Returns:
        読み込んだ3次元numpy配列。
    """
    return tiff.imread(input_path)


def create_execution_slice_mask(
    z_size: int, execution_slice_range: tuple | None = None
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """実行対象スライス範囲の真偽マスクを作成する。

    Args:
        z_size: Z方向のスライス数。
        execution_slice_range: 1-basedの(start, end)範囲。Noneなら全スライス対象。

    Returns:
        0-basedで適用可能なboolマスク配列と、正規化済みの1-based範囲。
    """
    mask = np.ones(z_size, dtype=bool)
    normalized_range = None
    if execution_slice_range is None:
        return mask, normalized_range

    if len(execution_slice_range) != 2:
        raise ValueError(
            f"execution_slice_range must be (start, end): {execution_slice_range}"
        )

    start, end = int(execution_slice_range[0]), int(execution_slice_range[1])
    start = max(1, start)
    end = min(z_size, end)
    if start > end:
        raise ValueError(f"No valid slice range after clamping: {execution_slice_range}")

    mask[:] = False
    mask[start - 1 : end] = True
    normalized_range = (start, end)
    return mask, normalized_range


def apply_execution_slice_mask(volume: np.ndarray, execution_mask: np.ndarray) -> np.ndarray:
    """Z方向マスクで対象外スライスを0化する。

    Args:
        volume: 入力3次元ボリューム。
        execution_mask: Z方向に対応したboolマスク。

    Returns:
        対象外スライスを0にしたボリューム。
    """
    return np.where(execution_mask[:, None, None], volume, 0).astype(volume.dtype)


def save_json(data: dict, output_path: str) -> str:
    """辞書データをJSONファイルとして保存する。

    Args:
        data: 保存する辞書データ。
        output_path: 出力JSONファイルパス。

    Returns:
        保存先のファイルパス。
    """
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path
