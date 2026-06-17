import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi


def rodrigues_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """from_vec を to_vec に回す 3x3 回転行列を返す。"""
    a = from_vec / (np.linalg.norm(from_vec) + 1e-12)
    b = to_vec / (np.linalg.norm(to_vec) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        # 平行 or 逆平行
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180度回転: a と直交する軸を1つ作る
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis) + 1e-12
        # 180度回転行列: R = -I + 2uu^T
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    k = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3) + k + (k @ k) * ((1.0 - c) / (s * s))
    return r


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """2値マスクの表面ボクセルを抽出。"""
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)  # 6近傍
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    surface = mask & (~eroded)
    return surface


def choose_occlusal_candidates(
    surface_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    percentile: float,
    side: str,
) -> np.ndarray:
    """
    咬合面候補点を抽出。
    side:
      - high: z が大きい側を候補
      - low : z が小さい側を候補
      - auto: 点数が多い側を採用
    """
    idx = np.argwhere(surface_zyx)
    if idx.size == 0:
        raise ValueError("表面ボクセルが見つかりません。入力が空の可能性があります。")

    z = idx[:, 0]

    z_low_thr = np.percentile(z, percentile)          # 例: 85%
    z_high_thr = np.percentile(z, 100.0 - percentile)  # 例: 15%

    cand_high = idx[z >= z_low_thr]
    cand_low = idx[z <= z_high_thr]

    if side == "high":
        chosen = cand_high
    elif side == "low":
        chosen = cand_low
    else:
        chosen = cand_high if len(cand_high) >= len(cand_low) else cand_low

    if len(chosen) < 20:
        raise ValueError(
            f"咬合面候補点が少なすぎます: {len(chosen)} 点。"
            " percentile を下げるか side を変えてください。"
        )

    # 物理座標へ変換 (z,y,x) * spacing
    pts_mm = chosen.astype(np.float64) * spacing_zyx[None, :]
    return pts_mm


def fit_plane_normal_svd(points_mm: np.ndarray) -> np.ndarray:
    """SVDで平面法線（最小特異ベクトル）を求める。"""
    center = points_mm.mean(axis=0)
    x = points_mm - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal) + 1e-12
    return normal


def rotate_volume_to_occlusal_parallel(
    volume: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    percentile=85.0,
    side="auto",
    pad=20,
):
    """
    歯2値ボリュームを回転し、咬合面法線をスライス法線(z軸)に一致させる。
    戻り値:
      rotated_uint8, info(dict)
    """
    if volume.ndim != 3:
        raise ValueError(f"3次元配列を想定していますが shape={volume.shape}")

    mask = volume > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("前景が空です。2値化結果を確認してください。")

    # 端の切れを減らすためにパディング
    if pad > 0:
        mask = np.pad(mask, ((pad, pad), (pad, pad), (pad, pad)), mode="constant")
    shape = np.array(mask.shape, dtype=np.float64)

    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    if np.any(spacing_zyx <= 0):
        raise ValueError(f"spacing は正数である必要があります: {spacing_zyx}")

    surface = extract_surface(mask)
    pts_mm = choose_occlusal_candidates(surface, spacing_zyx, percentile, side)
    n = fit_plane_normal_svd(pts_mm)

    # 方向を +z 側へそろえる（向き反転を防ぐ）
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # z,y,x 系で +z
    if np.dot(n, target) < 0:
        n = -n

    # 物理座標系の回転
    r_phys = rodrigues_rotation_matrix(n, target)

    # index座標系への変換: A = S^-1 R S
    s = np.diag(spacing_zyx)
    s_inv = np.diag(1.0 / spacing_zyx)
    a = s_inv @ r_phys @ s

    # affine_transform は output -> input 変換を受け取るため逆行列を使う
    m = np.linalg.inv(a)
    center = (shape - 1.0) / 2.0
    offset = center - m @ center

    # 2値なので最近傍補間
    rotated = ndi.affine_transform(
        mask.astype(np.uint8),
        matrix=m,
        offset=offset,
        output_shape=tuple(mask.shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    rotated = (rotated > 0).astype(np.uint8) * 255

    # 参考情報
    cos_theta = float(np.clip(np.dot(n, target), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))

    info = {
        "input_shape_zyx": [int(v) for v in volume.shape],
        "working_shape_zyx": [int(v) for v in mask.shape],
        "spacing_zyx": [float(v) for v in spacing_zyx],
        "estimated_normal_zyx": [float(v) for v in n],
        "target_normal_zyx": [1.0, 0.0, 0.0],
        "rotation_angle_deg": theta_deg,
        "percentile": float(percentile),
        "side": side,
        "pad": int(pad),
    }
    return rotated, info


def main():
    parser = argparse.ArgumentParser(
        description="歯2値3Dボリュームを回転し、咬合面をスライス面に平行化する。"
    )
    parser.add_argument("--input", required=True, help="入力3D tif")
    parser.add_argument("--output", required=True, help="出力3D tif")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SZ", "SY", "SX"),
        help="ボクセル間隔 (z y x)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=85.0,
        help="咬合面候補抽出に使う端側パーセンタイル (50-99推奨)",
    )
    parser.add_argument(
        "--side",
        choices=["auto", "high", "low"],
        default="auto",
        help="咬合面候補を z高側/低側/自動で選択",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=20,
        help="回転前パディング幅",
    )
    parser.add_argument(
        "--info-json",
        default="",
        help="回転情報を書き出す json パス（省略可）",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)



if __name__ == "__main__":
    input_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\labeled_map_filled\label_map_8_filled.tif"
    output_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\label_map_8_filled_aligned.tif"
    spacing = [1.0, 1.0, 1.0]
    percentile = 95
    side = "auto"
    pad = 20
    info_json = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\label_map_8_filled_aligned_info.json"
    vol = tiff.imread(str(input_path))
    rotated, info = rotate_volume_to_occlusal_parallel(
        volume=vol,
        spacing_zyx=spacing,
        percentile=percentile,
        side=side,
        pad=pad,
    )

    tiff.imwrite(str(output_path), rotated.astype(np.uint8), dtype=np.uint8)

    if info_json:
        p = Path(info_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", output_path)
    print("Rotation angle [deg]:", f"{info['rotation_angle_deg']:.3f}")
    print("Estimated normal [z,y,x]:", info["estimated_normal_zyx"])
    
    # main()