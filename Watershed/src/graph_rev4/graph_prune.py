import networkx as nx


def find_red_edge_previous_node(graph: nx.Graph, start_node) -> bool:
    """指定ノード以下の経路に赤エッジが存在するかを探索する。

    Args:
        graph: 探索対象グラフ。
        start_node: 探索開始ノード。

    Returns:
        赤エッジが見つかった場合はTrue、見つからない場合はFalse。
    """
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph.neighbors(node):
            # Only traverse to lower/equal slices.
            if neighbor[0] > node[0]:
                continue
            if graph[node][neighbor].get("color") == "red":
                return True
            stack.append(neighbor)

    return False


def remove_edge(graph: nx.Graph) -> None:
    """赤エッジを判定して色変更またはseed設定を行う。

    Args:
        graph: 変更対象グラフ。

    Returns:
        なし。
    """
    for previous_node, current_node, attr in sorted(
        graph.edges(data=True),
        key=lambda e: max(e[0][0], e[1][0]),
        reverse=True,
    ):
        if attr.get("color") != "red":
            continue

        if find_red_edge_previous_node(graph, previous_node):
            graph.edges[current_node, previous_node]["color"] = "green"
        else:
            graph.nodes[previous_node]["seed"] = True
