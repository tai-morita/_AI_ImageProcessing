import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tifffile as tiff


def save_colorized_labels_local(labels: np.ndarray, color_out: str):
    """ラベル画像を疑似カラー化してTIFF保存する。

    Args:
        labels: ラベル値を持つ2D/3D配列。
        color_out: 出力TIFFファイルパス。

    Returns:
        なし。
    """
    max_label = int(labels.max())
    if max_label == 0:
        rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    else:
        cmap = plt.get_cmap("nipy_spectral", max_label + 1)
        normalized = labels.astype(np.float32) / max_label
        rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        rgb[labels == 0] = 0
    tiff.imwrite(color_out, rgb, photometric="rgb")


def plot_graph_for_slice_range(
    graph: nx.Graph,
    slice_range: tuple | None = None,
    output_path: str | None = None,
    title: str | None = None,
    show: bool = False,
):
    """指定スライス範囲のグラフを描画し、必要に応じて保存する。

    Args:
        graph: 描画対象グラフ。
        slice_range: 1-basedの描画対象スライス範囲。
        output_path: 保存先画像パス。Noneなら保存しない。
        title: グラフタイトル。
        show: Trueの場合は画面表示する。

    Returns:
        保存した場合は出力パス、描画対象がない場合はNone。
    """
    real_nodes = [
        node for node in graph.nodes if isinstance(node[1], (int, np.integer))
    ]
    if slice_range is not None:
        start, end = int(slice_range[0]), int(slice_range[1])
        real_nodes = [node for node in real_nodes if start <= node[0] <= end]

    slices_with_real_nodes = sorted({node[0] for node in real_nodes})
    if not slices_with_real_nodes:
        return None

    draw_nodes = set(real_nodes)
    draw_nodes.update(
        (slice_index, "slice_index")
        for slice_index in slices_with_real_nodes
        if (slice_index, "slice_index") in graph
    )
    draw_graph = graph.subgraph(draw_nodes).copy()

    min_slice = min(slices_with_real_nodes)
    max_slice = max(slices_with_real_nodes)
    label_values = [int(node[1]) for node in real_nodes]
    max_label = max(label_values) if label_values else 1

    pos = {}
    for node in draw_graph.nodes:
        slice_index, label = node
        if label == "slice_index":
            pos[node] = (0, slice_index)
        else:
            pos[node] = (int(label), slice_index)

    height = max(8, min(80, (max_slice - min_slice + 1) * 0.35))
    width = max(10, min(36, max_label * 0.9 + 3))
    plt.figure(figsize=(width, height))

    labels = {}
    for node in draw_graph.nodes:
        if node[1] == "slice_index":
            labels[node] = f"slice {node[0]}"
        else:
            labels[node] = f"{draw_graph.nodes[node].get('area', '')}"

    node_colors = [
        "red" if draw_graph.nodes[node].get("seed", False) else "skyblue"
        for node in draw_graph.nodes
    ]
    edge_colors = [draw_graph[u][v].get("color", "black") for u, v in draw_graph.edges]

    nx.draw(
        draw_graph,
        pos=pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=450,
        font_size=7,
    )
    if title is not None:
        plt.title(title)
    plt.ylim(min_slice - 1, max_slice + 1)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return output_path
