import networkx as nx
import numpy as np
from scipy import ndimage as ndi


def create_graph(
    volume: np.ndarray,
    threshold: int,
    labels_connectivity: int = 2,
) -> tuple[nx.Graph, np.ndarray]:
    """隣接スライス間の重なりに基づく重み付きグラフを構築する。

    Args:
        volume: 入力3次元ボリューム。
        threshold: 赤エッジ判定に使う面積閾値。
        labels_connectivity: 2D連結成分ラベリングの近傍設定。

    Returns:
        構築済みグラフとスライスごとのラベルボリューム。
    """
    graph = nx.Graph()
    labeled_volume = np.zeros_like(volume, dtype=np.uint16)
    structure = ndi.generate_binary_structure(rank=2, connectivity=labels_connectivity)

    for index, curr_slice in enumerate(volume):
        slice_index = index + 1
        labels, labels_number = ndi.label(curr_slice, structure=structure)

        for label_number in range(1, labels_number + 1):
            label_area = int(np.sum(labels == label_number))
            graph.add_node((slice_index, label_number), area=label_area, seed=False)

        # Always keep labeled slices, including first slice.
        labeled_volume[index] = labels.astype(np.uint16)

        if slice_index != 1:
            prev_slice_index = slice_index - 1
            prev_slice = volume[prev_slice_index - 1]
            prev_labels, _ = ndi.label(prev_slice, structure=structure)

            for label_number in range(1, labels_number + 1):
                overlaps = np.unique(prev_labels[labels == label_number])
                overlaps = overlaps[overlaps != 0]

                edge_color = "blue"
                edge_color_count = 0
                for prev_label_number in overlaps:
                    if len(overlaps) >= 2 and threshold <= graph.nodes[
                        (prev_slice_index, int(prev_label_number))
                    ]["area"]:
                        edge_color_count += 1
                if len(overlaps) > 0 and edge_color_count >= len(overlaps):
                    edge_color = "red"

                for prev_label_number in overlaps:
                    prev_label_number = int(prev_label_number)
                    src = (slice_index, label_number)
                    dst = (prev_slice_index, prev_label_number)
                    if src in graph and dst in graph:
                        graph.add_edge(src, dst, color=edge_color)

        graph.add_node((slice_index, "slice_index"), seed=False)

    return graph, labeled_volume


def merge_bidirectional_graph(
    forward_graph: nx.Graph,
    reverse_graph: nx.Graph,
    num_slices: int,
) -> nx.Graph:
    """順方向グラフと逆方向グラフを同一座標系に統合する。

    Args:
        forward_graph: 順方向で構築したグラフ。
        reverse_graph: 逆方向で構築したグラフ。
        num_slices: 全スライス数。

    Returns:
        色優先度を反映した統合グラフ。
    """
    merged_graph = nx.Graph()
    color_priority = {"black": 0, "blue": 1, "green": 2, "red": 3}

    def upsert_node(node, attrs):
        if node not in merged_graph:
            merged_graph.add_node(node, **attrs)
            return
        merged_attrs = merged_graph.nodes[node]
        merged_attrs["seed"] = merged_attrs.get("seed", False) or attrs.get("seed", False)
        if "area" in attrs:
            merged_attrs["area"] = max(
                int(merged_attrs.get("area", 0)),
                int(attrs["area"]),
            )

    def upsert_edge(node_u, node_v, attrs):
        new_color = attrs.get("color", "black")
        if merged_graph.has_edge(node_u, node_v):
            current_color = merged_graph[node_u][node_v].get("color", "black")
            if color_priority.get(new_color, 0) > color_priority.get(current_color, 0):
                merged_graph[node_u][node_v]["color"] = new_color
            return
        merged_graph.add_edge(node_u, node_v, color=new_color)

    for node, attrs in forward_graph.nodes(data=True):
        upsert_node(node, dict(attrs))
    for node_u, node_v, attrs in forward_graph.edges(data=True):
        upsert_edge(node_u, node_v, attrs)

    for node, attrs in reverse_graph.nodes(data=True):
        mapped = (num_slices - node[0] + 1, node[1])
        upsert_node(mapped, dict(attrs))
    for node_u, node_v, attrs in reverse_graph.edges(data=True):
        mapped_u = (num_slices - node_u[0] + 1, node_u[1])
        mapped_v = (num_slices - node_v[0] + 1, node_v[1])
        upsert_edge(mapped_u, mapped_v, attrs)

    return merged_graph


def map_reverse_graph_to_original(reverse_graph: nx.Graph, num_slices: int) -> nx.Graph:
    """逆方向グラフを元のスライス座標系へ写像する。

    Args:
        reverse_graph: 逆順ボリューム由来のグラフ。
        num_slices: 元ボリュームの全スライス数。

    Returns:
        元の座標系に変換されたグラフ。
    """
    mapped = nx.Graph()
    for node, attrs in reverse_graph.nodes(data=True):
        mapped_node = (num_slices - node[0] + 1, node[1])
        mapped.add_node(mapped_node, **dict(attrs))
    for node_u, node_v, attrs in reverse_graph.edges(data=True):
        mapped_u = (num_slices - node_u[0] + 1, node_u[1])
        mapped_v = (num_slices - node_v[0] + 1, node_v[1])
        mapped.add_edge(mapped_u, mapped_v, color=attrs.get("color", "black"))
    return mapped
