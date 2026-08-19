# Root Canal を Seed にして Watershed を実行するテスト
import numpy as np
from ..WaterShed_main import watershed_3d_tiff

def edit_root_canal(labeled_root_canal: np.ndarray) -> np.ndarray:
    """
    ラベル付きの根管領域を編集する。
    0 を背景として除外し、ラベルごとのサイズ分布に大きな飛びがある箇所を境界として外れ値を除去する。
    """
    unique_labels, counts = np.unique(labeled_root_canal, return_counts=True)

    # 背景 0 は除外
    mask = unique_labels != 0
    unique_labels = unique_labels[mask]
    counts = counts[mask]

    if unique_labels.size == 0:
        return labeled_root_canal

    # サイズを昇順にソート
    order = np.argsort(counts)
    unique_labels = unique_labels[order]
    counts = counts[order]

    # 対数変換して、急なジャンプを見つける
    log_counts = np.log10(counts.astype(np.float64))
    jumps = np.diff(log_counts)

    if jumps.size == 0:
        return labeled_root_canal

    # 最大ジャンプ位置を境界として採用
    boundary_idx = np.argmax(jumps)
    threshold = counts[boundary_idx + 1]

    # 閾値未満のラベルは背景化
    valid_labels = unique_labels[counts >= threshold]
    edited_root_canal = np.zeros_like(labeled_root_canal)
    for label in valid_labels:
        edited_root_canal[labeled_root_canal == label] = label

    return relabel_without_zero(edited_root_canal)

# ラベル数が 0 のものを除外して、ラベルを 1 から順に振り直す
def relabel_without_zero(arr: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(arr)
    unique_labels = unique_labels[unique_labels != 0]  # 0 を除外

    for new_label, old_label in enumerate(unique_labels, start=1):
        arr[arr == old_label] = new_label

    return arr

def main() -> None:
    import tifffile
    input_path = r"D:\_study\ImageProcessing\study\Watershed\temp\CTHRs_50_label_binary_labeled_root_canal.tif"
    volume = tifffile.imread(input_path)
    edited_volume = edit_root_canal(volume)
    edited_output_path = r"D:\_study\ImageProcessing\study\Watershed\temp\CTHRs_50_label_binary_labeled_root_canal_edited.tif"
    tifffile.imwrite(edited_output_path, edited_volume)
    volume_path = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\CTHRs_50_label_binary.tif"
    watershed_3d_tiff(
        input_path   = volume_path,
        markers_path = edited_output_path,
        labels_out   = r"D:\_study\ImageProcessing\study\Watershed\t-oe\20260716_NR_Label\testWS\watershed_Tooth_label_conn=26.tif",
        connectivity = 26
    )
    return

if __name__ == "__main__":
    main()