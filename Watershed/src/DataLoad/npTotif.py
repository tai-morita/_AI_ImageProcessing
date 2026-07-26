# npy 形式のファイルを tif に変換して保存する
import numpy as np
import tifffile
import os

def npy_to_tif(npy_file, tif_file):
    # npy ファイルを読み込む
    data = np.load(npy_file)
    print(np.mean(data))
    # tif ファイルとして保存する
    tifffile.imwrite(tif_file, data)

if __name__ == "__main__":
    npy_file = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\CTHRs_50.npy"  # 変換したい npy ファイルのパス
    tif_file = os.path.splitext(npy_file)[0] + ".tif"  # 保存する tif ファイルのパス
    npy_to_tif(npy_file, tif_file)
    print(f"{npy_file} を {tif_file} に変換しました。")