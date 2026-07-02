import numpy as np
import os
import sys

try:
    from ..RodriguesRotation import rotate_volume_to_occlusal_parallel
except ImportError:
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from RodriguesRotation import rotate_volume_to_occlusal_parallel


def align_occlusal_plane(
    volume: np.ndarray,
    enabled: bool = True,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile: float = 85.0,
    side: str = "auto",
    pad: int = 0,
    inplane_k45: int = 0,
) -> tuple[np.ndarray, dict]:
    """咬合平面を基準軸に合わせる前処理を実行する。

    Args:
        volume: 入力3次元ボリューム。
        enabled: Trueの場合のみ咬合平面合わせを実行。
        spacing_zyx: Z, Y, X の画素間隔。
        percentile: 咬合面候補抽出に使う百分位。
        side: 咬合面候補の選択方向。
        pad: 回転前の余白パディング量。
        inplane_k45: 面内回転の45度単位指定。

    Returns:
        回転後ボリュームと実行情報辞書。
    """
    if not enabled:
        return volume, {"enabled": False}

    rotated, align_info = rotate_volume_to_occlusal_parallel(
        volume=volume,
        spacing_zyx=spacing_zyx,
        percentile=percentile,
        side=side,
        pad=pad,
        inplane_k45=inplane_k45,
    )
    rotated = np.asarray(rotated)
    if volume.dtype != rotated.dtype:
        rotated = rotated.astype(volume.dtype)

    info = {
        "enabled": True,
        "spacing_zyx": list(spacing_zyx),
        "percentile": float(percentile),
        "side": side,
        "pad": int(pad),
        "inplane_k45": int(inplane_k45),
        "detail": align_info,
    }
    return rotated, info
