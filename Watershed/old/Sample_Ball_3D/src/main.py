from . import test_watershed_vincent_soille
import numpy as np
import tifffile
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 小さなテスト画像（2つの極小を持つ人工地形）
    img = tifffile.imread(r"./study/Watershed/Sample_Ball_3D/input/spheres.tif")
    img = img[143]
    labels_ws = test_watershed_vincent_soille(img, connectivity=8, keep_watershed=True)
    labels_fill = test_watershed_vincent_soille(img, connectivity=8, keep_watershed=False)

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")
    axes[1].imshow(labels_ws, cmap="tab20")
    axes[1].set_title("Watershed (keep WSHED)")
    axes[2].imshow(labels_fill, cmap="tab20")
    axes[2].set_title("Watershed (fill WSHED)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(r"./study/Watershed/Sample_Ball_3D/output/watershed3d_spheres.png")