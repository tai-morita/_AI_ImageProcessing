from __future__ import annotations
from typing import Literal, Tuple, Dict, Any
import numpy as np
from skimage import io, color, exposure, filters, morphology, util
from scipy import ndimage as ndi
import os
import numpy as np

def generate_height_map_5basins(H=100, W=100, seed=42):
    """
    5つの谷（局所最小）と複雑な稜線を持つ 100x100 高さマップを生成する。
    出力は 0..255 の uint8 スケールで、値が小さいほど谷（低地）。

    Returns
    -------
    height_map : np.ndarray shape (H, W), dtype=uint8
    seeds_idx  : list[(y, x)]  # 5個の種のピクセル座標
    markers    : np.ndarray shape (H, W), dtype=int32  # 0=未割当, 1..5=各種
    """
    rng = np.random.default_rng(seed)

    # 正規化座標（-1..1）
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx - (W - 1) / 2) / ((W - 1) / 2)
    y = (yy - (H - 1) / 2) / ((H - 1) / 2)

    # 5つの谷（ガウス盆地）: (cx, cy, sx, sy, weight)
    basins = [
        (-0.65, -0.55, 0.22, 0.18, 1.00),  # 左上
        ( 0.60, -0.60, 0.20, 0.20, 1.00),  # 右上
        (-0.70,  0.50, 0.18, 0.22, 1.00),  # 左下
        ( 0.70,  0.55, 0.20, 0.18, 1.00),  # 右下
        ( 0.00,  0.05, 0.25, 0.25, 1.20),  # 中央（少し深め）
    ]

    # 盆地（低くするので負方向に寄与）
    valleys = np.zeros((H, W), dtype=np.float64)
    seed_pixels = []
    for (cx, cy, sx, sy, w) in basins:
        g = np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))
        valleys -= w * g
        # 種の座標（最近傍ピクセルへ丸め）
        px = int(round((cx * ((W - 1) / 2)) + (W - 1) / 2))
        py = int(round((cy * ((H - 1) / 2)) + (H - 1) / 2))
        px = min(max(px, 0), W - 1)
        py = min(max(py, 0), H - 1)
        seed_pixels.append((py, px))

    # 稜線（サイン波ベースの尾根をいくつか重ねる：高くする）
    ridges = (
        0.18 * np.sin(3.0 * np.pi * x) +
        0.18 * np.sin(3.0 * np.pi * y) +
        0.10 * np.sin(5.0 * np.pi * (x + y)) +
        0.08 * np.sin(4.0 * np.pi * (x - 0.3)) * np.sin(4.0 * np.pi * (y + 0.1))
    )

    # サドル状の壁（中央付近に縦横のゆるい障壁をつくる：高くする）
    walls = (
        0.35 * np.exp(-(x / 0.18) ** 2) +   # 縦の壁
        0.25 * np.exp(-((y - 0.15) / 0.22) ** 2)  # 横の壁（少し上にずらす）
    )

    # 微少ノイズ（平坦化のために軽く平滑化）
    noise = rng.normal(loc=0.0, scale=0.02, size=(H, W))

    # 3x3 ボックスフィルタでノイズを滑らかに（簡易ぼかし）
    pad = np.pad(noise, 1, mode='reflect')
    blur = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
        pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
        pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    noise_smooth = 0.5 * noise + 0.5 * blur  # 原ノイズと平滑のブレンド
    noise_smooth *= 0.4  # 強度調整

    # 合成（ベースは0、谷はマイナス、稜線・壁はプラス）
    height = valleys + ridges + walls + noise_smooth

    # 正規化して 0..255 の整数へ（低いほど谷）
    height = height - height.min()
    height = height / (height.max() + 1e-12)
    height_u8 = (255.0 * height).astype(np.uint8)

    # マーカー（5つの種）
    markers = np.zeros((H, W), dtype=np.int32)
    for i, (py, px) in enumerate(seed_pixels, start=1):
        markers[py, px] = i

    return height_u8, seed_pixels, markers


def load_and_prepare_image(
    path: str,
    *,
    to_float: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    画像を読み込み、グレースケール化して 2D 配列として返す。

    Parameters
    ----------
    path : str
        画像ファイルパス。
    to_float : bool
        True の場合 [0,1] float32 に正規化して返す。False の場合元の dtype を維持。
    normalize : bool
        ヒストグラムの飽和除去（1%）でコントラスト補正する（exposure.rescale_intensity）。

    Returns
    -------
    img_gray : np.ndarray (H, W)
        グレースケール画像。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"画像が見つかりません: {path}")

    img = io.imread(path)

    # グレースケール化（RGB / RGBA / マルチチャンネル対応）
    if img.ndim == 3:
        # 例：RGBA の場合は RGB に落としてから gray
        if img.shape[2] == 4:
            img = util.img_as_float32(img)
            img = img[..., :3]  # alpha drop
        img_gray = color.rgb2gray(img)  # 返りは float64 [0,1]
        img_gray = img_gray.astype(np.float32)
    elif img.ndim == 2:
        # そのまま
        img_gray = img
        # 型次第で 0..1 へ変換
        if img_gray.dtype != np.float32 and img_gray.dtype != np.float64:
            img_gray = util.img_as_float32(img_gray)
    else:
        raise ValueError(f"想定外の画像次元: {img.shape}")

    # コントラスト正規化（ヒストグラムの1%を飽和切り）
    if normalize:
        p2, p98 = np.percentile(img_gray, (1, 99))
        if p98 > p2:
            img_gray = exposure.rescale_intensity(img_gray, in_range=(p2, p98))

    if not to_float:
        # 0..255 uint8 に戻す場合
        img_gray = (np.clip(img_gray, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    return img_gray


def morphology_denoise(
    img_gray: np.ndarray,
    *,
    method: Literal["opening", "closing", "median"] = "opening",
    selem_radius: int = 2,
) -> np.ndarray:
    """
    モルフォロジーによるノイズ除去。

    Parameters
    ----------
    img_gray : np.ndarray
        入力グレースケール（推奨: 0..1 float）。
    method : {"opening", "closing", "median"}
        - "opening": 小さな白ノイズ除去（背景黒/前景白を想定）
        - "closing": 黒ノイズの穴埋め（背景白/前景黒に強い）
        - "median": 形状保持のノイズ低減（非線形フィルタ）
    selem_radius : int
        構造要素（disk）の半径（median の場合はサイズ計算に利用）。

    Returns
    -------
    img_denoise : np.ndarray
        ノイズ除去後の画像（dtype は入力に準拠）。
    """
    img = img_gray
    is_float = np.issubdtype(img.dtype, np.floating)

    # 形態学演算は [0,1] が都合よいので一時的に変換
    work = util.img_as_float32(img) if not is_float else img.astype(np.float32)

    selem = morphology.disk(selem_radius)

    if method == "opening":
        # 膨張(white)に弱いソルトノイズ除去
        # skimage の opening はグレースケールにも適用可
        work = morphology.opening(work, selem)
    elif method == "closing":
        work = morphology.closing(work, selem)
    elif method == "median":
        work = filters.median(work, footprint=selem)
    else:
        raise ValueError("method は 'opening' | 'closing' | 'median'")

    # dtype を元に合わせる
    if not is_float:
        work = util.img_as_ubyte(np.clip(work, 0, 1))
    return work


def segment_foreground_background(
    img_gray: np.ndarray,
    *,
    otsu_offset: float = 0.0,
    invert: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Otsu の大津の方法で前景/背景を二値化。

    Parameters
    ----------
    img_gray : np.ndarray (0..1 float 推奨)
    otsu_offset : float
        Otsu 閾値に加えるオフセット（例：0.02 で少し厳しく）。
    invert : bool
        True の場合、前景/背景を反転。

    Returns
    -------
    fg_mask : np.ndarray (bool)
        True = 前景
    threshold : float
        使用した実際の閾値
    """
    img = img_gray.astype(np.float32) if not np.issubdtype(img_gray.dtype, np.floating) else img_gray
    t = filters.threshold_otsu(img)
    t = float(np.clip(t + otsu_offset, 0.0, 1.0))
    fg = img > t
    if invert:
        fg = ~fg
    return fg, t


def distance_transform_pair(
    fg_mask: np.ndarray,
    *,
    return_foreground_dist: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    背景→前景のユークリッド距離（標準）と、必要なら前景→背景の距離も返す。

    Parameters
    ----------
    fg_mask : np.ndarray (bool)
        True が前景。
    return_foreground_dist : bool
        True の場合、前景内部から背景までの距離も計算して返す。

    Returns
    -------
    dist_bg_to_fg : np.ndarray (float32)
        背景ピクセルが最近傍の前景まで何ピクセルか（ユークリッド距離）。
    dist_fg_to_bg : np.ndarray | None
        前景ピクセルが最近傍の背景までの距離（必要時のみ）。
    """
    # 背景ピクセルに対して EDT（背景 True で distance）をかけると
    # 最近傍の False までの距離になるため、~fg_mask を使う。
    dist_bg_to_fg = ndi.distance_transform_edt(~fg_mask).astype(np.float32)

    dist_fg_to_bg = None
    if return_foreground_dist:
        dist_fg_to_bg = ndi.distance_transform_edt(fg_mask).astype(np.float32)

    return dist_bg_to_fg, dist_fg_to_bg


def preprocess_image_to_distance(
    path: str,
    *,
    denoise_method: Literal["opening", "closing", "median"] = "opening",
    selem_radius: int = 2,
    otsu_offset: float = 0.0,
    invert_binary: bool = False,
    return_foreground_dist: bool = False,
    normalize_input: bool = True,
) -> Dict[str, Any]:
    """
    画像読み込み → ノイズ除去（モルフォロジー）→ Otsu 二値化 → 距離画像（EDT）
    までを実行し、2次元配列をまとめて返す高レベル関数。

    Returns
    -------
    {
      "gray": np.ndarray,           # 入力グレースケール（0..1 float）
      "denoised": np.ndarray,       # ノイズ除去後
      "fg_mask": np.ndarray,        # 前景マスク(bool)
      "threshold": float,           # 使用閾値（Otsu+offset）
      "dist_bg_to_fg": np.ndarray,  # 背景→前景の距離（float32）
      "dist_fg_to_bg": np.ndarray | None  # 前景→背景の距離（要求時のみ）
    }
    """
    gray = load_and_prepare_image(path, to_float=True, normalize=normalize_input)
    denoised = morphology_denoise(gray, method=denoise_method, selem_radius=selem_radius)
    # denoised は uint8 の可能性があるので 0..1 float に統一
    denoised_f = denoised.astype(np.float32) / (255.0 if denoised.dtype == np.uint8 else 1.0)

    fg_mask, thr = segment_foreground_background(denoised_f, otsu_offset=otsu_offset, invert=invert_binary)
    dist_bg_to_fg, dist_fg_to_bg = distance_transform_pair(fg_mask, return_foreground_dist=return_foreground_dist)

    return {
        "gray": gray.astype(np.float32),
        "denoised": denoised_f.astype(np.float32),
        "fg_mask": fg_mask.astype(bool),
        "threshold": float(thr),
        "dist_bg_to_fg": dist_bg_to_fg,
        "dist_fg_to_bg": dist_fg_to_bg,
    }

def translate_gray(img):
    # 画像をグレースケール化する
    arr2d_img = io.imread(img, as_gray=True)
    if arr2d_img.ndim == 3:
        img_gray = color.rgb2gray(arr2d_img)
    elif arr2d_img.ndim == 2:
        img_gray = arr2d_img
    else:
        raise ValueError("想定外の画像次元: {}".format(arr2d_img.shape))
    save_path = './study/test/Output/coin_gray.csv'
    np.savetxt(save_path, img_gray, delimiter=',', fmt='%.4f')
    # 画像で保存する
    img_save_path = './study/test/Output/coin_gray.png'
    io.imsave(img_save_path, (img_gray * 255).astype(np.uint8))
    return img_gray

def get_seed_coords(img):
    # 画像からseed点の座標を取得する
    # グレースケール画像
    arr2d_img = io.imread(img, as_gray=True)
    np.savetxt('./study/test/Output/preprocess_gray.csv', arr2d_img, delimiter=',', fmt='%.4f')

if __name__ == "__main__":
    
    out = preprocess_image_to_distance(
            r"./study/test/Input/water_coins.jpg",
            denoise_method="opening",    # "opening" | "closing" | "median"
            selem_radius=2,
            otsu_offset=0.0,
            invert_binary=False,         # 背景/前景の極性が逆なら True
            return_foreground_dist=True, # 前景→背景の距離も欲しい場合
            normalize_input=True,
        )

    # np.savetxt('./study/test/Output/preprocess_gray.csv', out["gray"], delimiter=',', fmt='%.4f')

    get_seed_coords(r"./study/test/Input/marked_coin_gray.png")

    # gray = out["gray"]
    # denoised = out["denoised"]
    # fg_mask = out["fg_mask"]
    # dist_bg_to_fg = out["dist_bg_to_fg"]
    # dist_fg_to_bg = out["dist_fg_to_bg"]



    # 可視化例（任意）
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    # ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Gray")
    # ax[1].imshow(denoised, cmap="gray"); ax[1].set_title("Denoised")
    # ax[2].imshow(fg_mask, cmap="gray"); ax[2].set_title("Foreground (Otsu)")
    # im = ax[3].imshow(dist_bg_to_fg, cmap="magma"); ax[3].set_title("EDT (BG→FG)")
    # fig.colorbar(im, ax=ax[3])
    # [a.axis("off") for a in ax]
    # plt.tight_layout(); plt.show()
