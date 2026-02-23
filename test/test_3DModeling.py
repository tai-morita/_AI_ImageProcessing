import numpy as np
import tifffile as tiff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 3D tiffファイルを読み込み、ボリュームデータを可視化するサンプル

def visualize_3d_tiff(tiff_path, threshold=None):
    # TIFF読み込み
    vol = tiff.imread(tiff_path)
    if vol.ndim != 3:
        raise ValueError(f"3Dボリュームを想定していますが、形状が {vol.shape} です。")

    # 前処理: 正規化
    vol = (vol - vol.min()) / (vol.max() - vol.min())

    # しきい値で前景抽出（任意）
    if threshold is None:
        threshold = 0.5
    mask = vol > threshold

    # 3D座標取得
    z, y, x = np.where(mask)

    # 3Dプロット
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Volume Visualization')
    plt.show()

if __name__ == "__main__":
    # サンプルパスを適宜変更
    visualize_3d_tiff("./Watershed/Tooth/data_053.tif")
