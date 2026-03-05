import numpy as np
import tifffile as tiff
import os

def axial_transpose(input_path, output_dir):
    # Axial 面を Sagittal, Coronal 面に変換して保存

    # 3D image: (Z, Y, X) または 4D image: (Z, Y, X, C)
    frames = tiff.imread(input_path)

    if frames.ndim == 3:
        # Sagittal: (X, Z, Y)
        frames_sagittal = np.transpose(frames, (2, 0, 1))
        # Coronal: (Y, Z, X)
        frames_coronal = np.transpose(frames, (1, 0, 2))
    elif frames.ndim == 4:
        # RGB等のチャネル付き: (Z, Y, X, C)
        # Sagittal: (X, Z, Y, C)
        frames_sagittal = np.transpose(frames, (2, 0, 1, 3))
        # Coronal: (Y, Z, X, C)
        frames_coronal = np.transpose(frames, (1, 0, 2, 3))
    else:
        raise ValueError(f"3D/4Dボリュームを想定していますが、shape={frames.shape} です。")
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名のみ取得
    base = os.path.splitext(os.path.basename(input_path))[0]

    tiff.imwrite(
        os.path.join(output_dir, f"{base}_sagittal.tif"),
        frames_sagittal
    )
    tiff.imwrite(
        os.path.join(output_dir, f"{base}_coronal.tif"),
        frames_coronal
    )

if __name__ == "__main__":
    axial_transpose(r"./Data/Input/data_053.tif",
                    r"./Data/Output")
    print("Done")