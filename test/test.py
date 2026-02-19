import os
import numpy as np
import tifffile as tiff
from skimage import filters, util
from scipy import ndimage as ndi
from PIL import Image

def to_uint16_per_slice(stack_float, clip_percentile=99.9):
    """
    3D配列 (Z, Y, X) の各スライスを独立に [0, clip] → [0, 65535] に線形スケール。
    距離画像の見やすいTIFF化用。値の相対関係はスライス内で保たれる。
    """
    z, h, w = stack_float.shape
    out = np.zeros((z, h, w), dtype=np.uint16)
    for i in range(z):
        sl = stack_float[i]
        # 0..clip_max にクリップ（外れ値への耐性）
        clip_max = np.percentile(sl, clip_percentile)
        clip_max = max(float(clip_max), 1e-12)
        sl_clip = np.clip(sl, 0, clip_max)
        out[i] = (sl_clip / clip_max * 65535).astype(np.uint16)
    return out

def threshold_main():
    in_path = r"./study/test/Input/yn-omusubi_inference_shrink4.tif"
    out_dir = r"./study/test/Output"
    os.makedirs(out_dir, exist_ok=True)

    # 1) 読み込み（(Z, Y, X) を想定。dtypeはuint8/uint16など）
    vol = tiff.imread(in_path)  # shape: (Z, Y, X)
    if vol.ndim != 3:
        raise ValueError(f"3Dスタック (Z,Y,X) を想定していますが、shape={vol.shape}")

    # 2) しきい値（二値化）と距離変換（各スライス2D）
    z, h, w = vol.shape
    binary_stack = np.zeros_like(vol, dtype=np.uint16)  # 0/65535 の2値で保存
    dist_stack = np.zeros((z, h, w), dtype=np.float32)  # 距離はfloatで保持→後でu16化
    thresholds = []

    for i in range(z):
        frame = vol[i]

        # Otsu しきい値（frame が16bitでもOK）
        # 明暗が逆なら > / < を反転してください
        thr = filters.threshold_otsu(frame)
        thresholds.append(float(thr))

        # 2値マスク（前景=True）
        bw = frame > thr

        # 保存用の2値（0/65535）
        binary_stack[i] = (bw.astype(np.uint16) * 65535)

        # 背景→前景までの距離（背景画素に距離が入る）
        dist_bg_to_fg = ndi.distance_transform_edt(~bw)
        dist_stack[i] = dist_bg_to_fg.astype(np.float32)

    # 3) 保存
    stem = os.path.splitext(os.path.basename(in_path))[0]

    # 2値マスク（0/65535）
    tiff.imwrite(
        os.path.join(out_dir, f"{stem}_binary_u16.tif"),
        binary_stack,
        photometric="minisblack"
    )

    # 距離（各スライス独立にスケールして16bit）
    dist_u16 = to_uint16_per_slice(dist_stack, clip_percentile=99.9)
    tiff.imwrite(
        os.path.join(out_dir, f"{stem}_distance_u16.tif"),
        dist_u16,
        photometric="minisblack"
    )

    # スライスごとのしきい値をログ保存（任意）
    with open(os.path.join(out_dir, f"{stem}_threshold_values.txt"), "w") as f:
        for i, thr in enumerate(thresholds):
            f.write(f"z={i}\totsu={thr}\n")

    print("done.")

def merge_Tooth():
    import glob
    dir = r"Watershed\Tooth\Data\data_53"
    data = glob.glob(os.path.join(dir, "*.png"))
    frames = []
    for d in data:
        img = Image.open(d).convert('L')  # グレースケールで読み込む
        frames.append(np.array(img))
    frames = np.stack(frames, axis=0)
    # 16bit TIFFで保存（0-65535の範囲にスケール）
    frames_u16 = to_uint16_per_slice(frames.astype(np.float32), clip_percentile=99.9)
    tiff.imwrite(os.path.join(dir, "data_53.tif"), frames_u16, photometric="minisblack")

if __name__ == "__main__":
    merge_Tooth()