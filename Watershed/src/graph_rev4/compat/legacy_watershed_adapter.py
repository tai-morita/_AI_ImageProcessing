import tifffile as tiff

from ..watershed_runner import watershed_3d_volume


def watershed_3d_tiff(
    input_path: str,
    markers_path: str,
    labels_out: str,
    connectivity: int = 6,
):
    """旧API互換でTIFF入力の3D watershedを実行する。

    Args:
        input_path: 入力ボリュームTIFFパス。
        markers_path: マーカーTIFFパス。Noneなら自動マーカー。
        labels_out: 出力ラベルTIFFパス。
        connectivity: 3次元連結性。

    Returns:
        生成したラベルボリューム。
    """
    volume = tiff.imread(input_path)
    markers = tiff.imread(markers_path) if markers_path is not None else None
    return watershed_3d_volume(
        volume=volume,
        seed_volume=markers,
        labels_out=labels_out,
        connectivity=connectivity,
    )
