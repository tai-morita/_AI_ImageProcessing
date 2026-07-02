import networkx as nx
import numpy as np


def detect_non_bifurcating_components(graph: nx.Graph):
    """二股分岐を含まない連結成分を抽出する。

    Args:
        graph: 判定対象グラフ。

    Returns:
        分岐を持たない連結成分ノード列のリスト。
    """
    valid_nodes = [
        node
        for node in graph.nodes
        if isinstance(node[1], (int, np.integer))
    ]
    subgraph = graph.subgraph(valid_nodes)

    non_bifurcating = []
    for component_nodes in nx.connected_components(subgraph):
        component_nodes = list(component_nodes)
        has_bifurcation = False

        for node in component_nodes:
            children = [
                neighbor
                for neighbor in subgraph.neighbors(node)
                if neighbor[0] > node[0]
            ]
            if len(children) >= 2:
                has_bifurcation = True
                break

        if not has_bifurcation:
            non_bifurcating.append(sorted(component_nodes, key=lambda n: (n[0], n[1])))

    return non_bifurcating


def add_largest_area_seed_in_components(graph: nx.Graph, components):
    """seed未設定成分に対して最大面積ノードをseedに追加する。

    Args:
        graph: 変更対象グラフ。
        components: 連結成分ごとのノード集合。

    Returns:
        新規にseed追加したノード一覧。
    """
    added_seed_nodes = []
    for component_nodes in components:
        if not component_nodes:
            continue

        has_seed = any(graph.nodes[node].get("seed", False) for node in component_nodes)
        if has_seed:
            continue

        target_node = max(component_nodes, key=lambda n: int(graph.nodes[n].get("area", 0)))
        graph.nodes[target_node]["seed"] = True
        added_seed_nodes.append(target_node)

    return added_seed_nodes
