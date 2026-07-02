import networkx as nx
import numpy as np


def create_seed_for_watershed(graph: nx.Graph, labeled_volume: np.ndarray) -> np.ndarray:
    """グラフ上のseedノードからwatershed用seedボリュームを作成する。

    Args:
        graph: seed属性を持つグラフ。
        labeled_volume: スライスごとの連結成分ラベルボリューム。

    Returns:
        連番seedラベルを持つ3次元ボリューム。
    """
    seed_volume = np.zeros_like(labeled_volume, dtype=labeled_volume.dtype)
    red_nodes = [node for node, attr in graph.nodes(data=True) if attr.get("seed", False)]

    next_seed_label = 1
    for slice_index, label_number in sorted(red_nodes, key=lambda node: (node[0], node[1])):
        if not isinstance(label_number, (int, np.integer)):
            continue
        if not (1 <= slice_index <= labeled_volume.shape[0]):
            continue

        label_number = int(label_number)
        slice_labels = labeled_volume[slice_index - 1]
        mask = slice_labels == label_number
        if not np.any(mask):
            continue
        seed_volume[slice_index - 1][mask] = next_seed_label
        next_seed_label += 1

    return seed_volume


def relabel_seed_volume(seed_volume: np.ndarray, start_label: int = 1):
    """seedラベルを指定開始番号から連番で再付番する。

    Args:
        seed_volume: 入力seedボリューム。
        start_label: 再付番の開始ラベル値。

    Returns:
        再ラベル後ボリュームと次に使うラベル値。
    """
    output = np.zeros_like(seed_volume, dtype=np.uint16)
    next_label = int(start_label)
    for label_value in np.unique(seed_volume):
        label_value = int(label_value)
        if label_value == 0:
            continue
        output[seed_volume == label_value] = next_label
        next_label += 1
    return output, next_label


def merge_seed_non_overlap(base_seed: np.ndarray, new_seed: np.ndarray) -> np.ndarray:
    """既存seedを優先して新規seedを非重複で統合する。

    Args:
        base_seed: 優先されるseedボリューム。
        new_seed: 空き領域へ反映するseedボリューム。

    Returns:
        統合後のseedボリューム。
    """
    return np.where(base_seed != 0, base_seed, new_seed).astype(np.uint16)
