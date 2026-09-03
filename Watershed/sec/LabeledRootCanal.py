import numpy as np
from scipy import ndimage as ndi

def labeled_for_root_canal(volume: np.ndarray, target_value: int = 1, dilation_iterations: int = 5) -> np.ndarray:
    """
    根管のラベル付け

    Parameters:
        volume (np.ndarray): 入力の3D画像データ
        target_value (int) : 抽出対象の値（現状 1）
        dilation_iterations (int): 膨張の反復回数（デフォルトは 5）

    Returns:
        np.ndarray: ラベル付きの3D画像データ

    Details:
        - volume 内の target_value の領域を抽出しラベリングをする
        - 周囲 5 ボクセルに対象があれば、同一のラベルとする
    """

    return label_nearby_regions(
        volume == target_value,
        dilation_iterations=dilation_iterations,
    )


def label_nearby_regions(
    mask: np.ndarray,
    dilation_iterations: int = 5, # 5 ボクセルまで膨張させてラベル付けする
) -> np.ndarray:
    """Expand nearby 3D regions, then label only the original foreground voxels."""
    # 隣接領域を膨張させてラベル付けする
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape}")
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    connectivity = ndi.generate_binary_structure(rank=3, connectivity=3)
    expanded_mask = ndi.binary_dilation(
        mask,
        structure=connectivity,
        iterations=dilation_iterations,
    )
    expanded_labels, _ = ndi.label(expanded_mask, structure=connectivity)
    return np.where(mask, expanded_labels, 0)


def relabel_nearby_labeled_regions(
    labeled_volume: np.ndarray,
    dilation_iterations: int = 5,
) -> np.ndarray:
    """Merge labels within the dilation range, then relabel them from 1."""
    if labeled_volume.ndim != 3:
        raise ValueError(f"Expected a 3D labeled_volume, got shape={labeled_volume.shape}")
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    labels = np.unique(labeled_volume)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros_like(labeled_volume, dtype=np.int32)

    parent = {int(label): int(label) for label in labels}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(label_a: int, label_b: int) -> None:
        root_a = find(label_a)
        root_b = find(label_b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    connectivity = ndi.generate_binary_structure(rank=3, connectivity=3)
    for label in labels:
        label = int(label)
        expanded_mask = ndi.binary_dilation(
            labeled_volume == label,
            structure=connectivity,
            iterations=dilation_iterations,
        )
        nearby_labels = np.unique(labeled_volume[expanded_mask])
        for nearby_label in nearby_labels:
            nearby_label = int(nearby_label)
            if nearby_label != 0:
                union(label, nearby_label)

    root_to_new_label: dict[int, int] = {}
    label_to_new_label: dict[int, int] = {}
    for label in labels:
        label = int(label)
        root = find(label)
        if root not in root_to_new_label:
            root_to_new_label[root] = len(root_to_new_label) + 1
        label_to_new_label[label] = root_to_new_label[root]

    relabeled_volume = np.zeros_like(labeled_volume, dtype=np.int32)
    for label, new_label in label_to_new_label.items():
        relabeled_volume[labeled_volume == label] = new_label
    return relabeled_volume
