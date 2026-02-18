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


if __name__ == "__main__":
    height_map, seeds, markers = generate_height_map_5basins(H=100, W=100, seed=2026)

    print("height_map shape:", height_map.shape, height_map.dtype)
    print("markers unique labels:", np.unique(markers))
    print("seeds (y, x):", seeds)

    # 必要なら可視化（コメント解除）
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(5,5))
    # plt.imshow(height_map, cmap="terrain")
    # ys, xs = zip(*seeds)
    # plt.scatter(xs, ys, c="red", s=30, marker="x", label="seeds")
    # plt.legend()
    # plt.title("Height Map (lower = darker)")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # ファイル保存例（必要な方を使用）
    # np.savetxt("height_map_100x100.csv", height_map, fmt="%d", delimiter=",")
    # np.save("height_map_100x100.npy", height_map)
    # np.savetxt("markers_100x100.csv", markers, fmt="%d", delimiter=",")