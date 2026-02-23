import numpy as np
import tifffile as tiff
import os

def axial_transpose(input_path, output_dir):
    # 3D image: (Z, Y, X)
    frames = tiff.imread(input_path)
    # Sagittal: (X, Z, Y)
    frames_sagittal = np.transpose(frames, (2, 0, 1))
    # Coronal: (Y, Z, X)
    frames_coronal = np.transpose(frames, (1, 0, 2))
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名のみ取得
    base = os.path.splitext(os.path.basename(input_path))[0]

    tiff.imwrite(
        os.path.join(output_dir, f"{base}_sagittal.tif"),
        frames_sagittal.astype(np.uint16)
    )
    tiff.imwrite(
        os.path.join(output_dir, f"{base}_coronal.tif"),
        frames_coronal.astype(np.uint16)
    )

if __name__ == "__main__":
    axial_transpose(r"./test\Input\data_053.tif",
                    r"./test\Input")
    axial_transpose(r"./test\Input\mask_edited.tif",
                    r"./test\Input")
    print("Done")