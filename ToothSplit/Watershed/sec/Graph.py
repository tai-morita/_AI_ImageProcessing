"""
1. グラフを作る
    1-0. volume の各スライスをラベル付けする
    1-1. (slice, label) の組み合わせでノードを作る(Connected Component のラベル)
        属性: 
            - area: 面積 (ボクセル数)
            - root_canal_label: 根管ラベル番号 (input)
            - inheritance_root_canal_label: 直前スライスから継承した根管ラベル番号 (input)
            - seed: (True/False)
    1-2. エッジを作る
        前後スライスが重なっているノードはエッジを引く
        ※ 根管ラベルの情報は重複して継承する
2. エッジの属性を付与する
    探索方向を決定する (up or down)
    探索方向の直前スライスのエッジの本数を確認する
        1 本であれば何もしない
        2 本以上であれば、根管ラベルを確認する
            異なる根管ラベルの場合は切断する (赤色にする)
            同じ根管ラベルの場合は切断しない (緑色にする)
    - color: エッジの色 (赤: 切断候補, 青: 切断しない, 緑: 切断候補だが下の層に赤エッジがあるので切断しない)
"""

from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .DataLoad import load_data, save_volume
from .LabeledRootCanal import labeled_for_root_canal
from .EditRootcanalLabel import edit_root_canal

# direction がある場合は、上から、下から作成していき、スライス内に追加するノードがない時に終了する。
def create_graph(volume: np.ndarray, 
                 labeled_volume_root_canal: np.ndarray, 
                 direction: str = "down",
                 labels_connectivity: int =2,
                 labeled_volume_per_slice: np.ndarray = None,
                 ) -> tuple[nx.Graph, np.ndarray]:
    if labeled_volume_per_slice is None:
        labeled_volume_per_slice = np.zeros_like(volume, dtype=np.int32)
    G = nx.Graph()
    count = 0

    # direction 用
    if direction == "down":
        direction_operator = -1
        slice_indices = range(volume.shape[0] - 1, -1, -1)
    elif direction == "up":
        direction_operator = 1
        slice_indices = range(volume.shape[0] - 1)

    continue_flag = True

    # 上顎・下顎の境目を検出する
    border_molar, _, _ = find_lowest_slice_number(volume)
    
    for index in slice_indices:
        curr_slice = volume[index]
        # スライスは1始まりにする
        slice_index = index + 1

        if False:
            min_slice, max_slice = 170, 300
            if slice_index not in range(min_slice, max_slice):
                continue

        # print(f"Processing slice {index + 1}/{volume.shape[0]}")
        count += 1

        """
        ndi.generate_binary_structure
        rank 次元の連結成分とみなす bool 値が入る
        rank=2, connectivity=1 なら 4 近傍なので
        False True False
        True  True True
        False True False
        """
        structure = ndi.generate_binary_structure(rank=2, connectivity=labels_connectivity)
        # structure 近傍を隣接画素としてラベル付けをする
        curr_labels, curr_label_numbers = ndi.label(curr_slice, structure=structure)
        labeled_volume_per_slice[index] = curr_labels

        # ラベル数が 0 の場合はグラフを作らない
        # 最初のスライスで 0 になる場合もあるので、その場合は次のスライスに進む
        # slice_index が border_molar に達した場合もグラフの作成を終了する
        if slice_index == border_molar:
            break
        if curr_label_numbers > 0 and continue_flag:
            # 0 以外のラベルが始まったら、もう終わってもいいので、 continue_flag を False にする
            continue_flag = False
        if curr_label_numbers == 0:
            if continue_flag:
                continue
            else:
                # 0 になったら終了する
                break


        for curr_label_number in range(1, curr_label_numbers+1):
            # 面積が属性となるため、ラベルごとに面積を計算する
            component_mask = (curr_labels == curr_label_number)
            label_area = np.sum(component_mask)

            # 根管のラベルを取得する
            root_canal_labels = get_root_canal_labels_for_component(
                component_mask=component_mask,
                labeled_volume_root_canal=labeled_volume_root_canal[index],
            )

            # 面積, 根管のラベルを属性としてノード(頂点)を追加
            G.add_node(
                (slice_index, curr_label_number),
                area=label_area,
                seed=False,
                root_canal_labels=root_canal_labels, # このスライスの根管ラベル
                inheritance_root_canal_label=root_canal_labels # 直前スライスから継承した根管ラベル
            )

        if slice_index == 1 or count == 1:
            # 最初のスライスは直前のスライスがなくエッジが引けない
            continue

        # 直前のスライスと比較してエッジ(辺)を引く
        prev_array_index = index - 1 * direction_operator
        prev_slice_index = slice_index - 1 * direction_operator
        prev_slice = volume[prev_array_index]
        prev_labels, prev_label_numbers = ndi.label(prev_slice, structure=structure)

        # 直前のスライスと重なっているところ = 同じ歯が表示されているとみなしてエッジを引く
        for curr_label_number in range(1, curr_label_numbers+1):
            # 現在のラベル番号が、 1 つ前のスライスの座標ではどのラベルがいるかを確認
            """
            Prev      Curr
            1, 2, 1   1, 3, 5
            2, 4, 0   1, 2, 1
            の場合、 Curr の 1 は Prev の 0, 1, 2 が対応する -> 0 は背景なので、 1, 2が隣接している
            """
            # overlap_label_numbers は prev_slice のラベル番号で、curr_slice の label_number のある座標に存在する番号を保持している
            # 上の例では、 label_number = 1 のとき overlap_label_numbers(1) = [1, 2] となる
            overlap_label_numbers = np.unique(prev_labels[curr_labels == curr_label_number])
            overlap_label_numbers = overlap_label_numbers[overlap_label_numbers != 0]

            # 重なっているノードでエッジを引く
            for overlap_label_number in overlap_label_numbers:
                overlap_label_number = int(overlap_label_number)
                if ((slice_index, curr_label_number) in G)\
                and ((prev_slice_index, overlap_label_number)) in G:
                    G.add_edge(
                        (slice_index, curr_label_number),
                        (prev_slice_index, overlap_label_number),
                        color = "blue"
                    )
                    # 根管ラベルは、前スライスの根管ラベルを追加する
                    prev_root_canal_labels = G.nodes[(prev_slice_index, overlap_label_number)]["inheritance_root_canal_label"]
                    root_canal_labels = G.nodes[(slice_index, curr_label_number)]["inheritance_root_canal_label"]
                    root_canal_labels.extend(prev_root_canal_labels)
                    G.nodes[(slice_index, curr_label_number)]["inheritance_root_canal_label"] = list(dict.fromkeys(root_canal_labels))
        # 表示用
        # 一番左にスライス番号を表示する
        G.add_node((slice_index, 'slice_index'), seed=False)

    return G, labeled_volume_per_slice

def get_root_canal_labels_for_component(component_mask: np.ndarray, labeled_volume_root_canal: np.ndarray) -> list[int]:
    """
    component_mask: ある connected component の True/False のマスク画像 (2D)
    labeled_volume_root_canal: root canal のラベル画像 (背景は 0)

    Returns:
        component_mask と重なっている root canal ラベルの一覧
        例: [3] or [3, 7] or []
    """
    # component_mask と labeled_volume_rootcanal の形状が一致しているか確認する
    if component_mask.shape != labeled_volume_root_canal.shape:
        raise ValueError(f"component_mask と labeled_volume_root_canal の形状が一致しません: component_mask.shape={component_mask.shape}, labeled_volume_root_canal.shape={labeled_volume_root_canal.shape}")
    # 成分と重なる根管ラベルがなければ、空のリストを返す
    overlapped = np.unique(labeled_volume_root_canal[component_mask])
    overlapped = overlapped[overlapped != 0]

    return overlapped.astype(int).tolist()


# グラフの描画
def plot_graph(G: nx, save_path: str = None):
    pos = {}
    for node in G.nodes:
        slice_index, label = node
        # スライス番号を左端に表示する
        if label == 'slice_index':
            pos[node] = (0, slice_index)
        else:
            pos[node] = (label, slice_index) # x:スライス順, y:ラベル番号で上下にずらす

    # plt.figure(figsize=(12, 20))
    slice_numbers = [node[0] for node in G.nodes]
    label_numbers = [
        node[1]
        for node in G.nodes
        if node[1] != "slice_index"
    ]

    slice_count = len(set(slice_numbers))
    max_labels_per_slice = max(label_numbers, default=1)

    figure_width = max(12, max_labels_per_slice * 1.5)
    figure_height = max(8, slice_count * 0.6)

    plt.figure(figsize=(figure_width, figure_height))

    # 描画するものを決める
    labels = {}
    for node in G.nodes:
        # スライス番号表示用のノードとメインのノードで表示するものを分ける
        if node[1] == 'slice_index':
            labels[node] = f"slice {node[0]}"
        else:
            # ノードのラベルは面積と根管のラベルを表示する
            labels[node] = f"{G.nodes[node]['area']}\n{G.nodes[node]['inheritance_root_canal_label']}"
            continue

    # ノードの色を決める。 Seed は赤く表示する
    node_colors = []
    for node in G.nodes:
        if G.nodes[node].get("seed", True):
            node_colors.append("red")
        else:
            node_colors.append('skyblue')

    # 色のついていないエッジは黒で表示する
    edge_colors = [
        G[src_node][tar_node].get("color", "black")
        for src_node, tar_node in G.edges
    ]

    nx.draw(G, pos=pos, 
            with_labels=True, 
            labels=labels, 
            node_color=node_colors,
            edge_color=edge_colors)


    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graph saved to {save_path}")


def edge_cut(G: nx.Graph, direction: str = "down") -> nx.Graph:
    """
    エッジの切断候補を決定する
    エッジの切断候補は赤になっているので、切断条件を満たさない場合は緑にする。
    切断判定:
        - 前後スライスでのノードが重なっている
        - 重なっているノードの根管ラベルが異なる場合は切断する
        - 切断候補以下のエッジで赤いエッジがある場合は切断しない (緑にする)
            - 例1:
            - node: slice 1: A, B, slice 2: C
            - edge: (A, C), (B, C)
            - (A, C), (B, C) どちらも切断候補
            - 例2:
            - node: slice 1: A, B, slice 2: C, D, slice3: E
            - edge: (A, C), (B, C), (C, E), (D, E)
            - (A, C), (B, C), (D, E) が切断候補
            - (C, E) は切断候補だが、下の層に赤いエッジがあるので切断しない (緑にする)
    Parameters:
        G: nx.Graph
        direction: "down" or "up"
            - "down": 上から下に探索する slice_index が大きいものから処理する
            - "up": 下から上に探索する slice_index が小さいものから処理する
    """
    if direction not in ["down", "up"]:
        raise ValueError(f"direction は 'down' または 'up' である必要があります。 direction={direction}")
    # reverse: True は slice_index の降順、 False は slice_index の昇順
    elif direction == "down":
        reverse = True
    elif direction == "up":
        reverse = False

    # ノードのないスライス番号を取得する
    slice_numbers = [node[0] for node in G.nodes]
    slice_numbers = list(set(slice_numbers))

    # ノードをスライス番号順に操作していき、エッジを切断するかどうか判定する
    for node, attr in sorted(G.nodes(data=True), key=lambda x: x[0][0], reverse=reverse):
        # スライス番号表示用のノードは無視する
        if node[1] == "slice_index":
            continue

        # スライス番号以下(以上)の隣接ノードを取得する
        neighbors = list(G.neighbors(node))
        if direction == "down":
            # スライス番号以上の隣接ノードのみを残す
            neighbors = [n for n in neighbors if n[0] > node[0]]
        elif direction == "up":
            # スライス番号以下の隣接ノードのみを残す
            neighbors = [n for n in neighbors if n[0] < node[0]]
        if len(neighbors) <= 1:
            # 隣接ノードが 1 つ以下の場合は切断判定を行わない
            continue

        # 根管ラベルが異なる場合は切断候補 (赤) にする
        # 根管ラベルを比較し、同じか、根管ラベルがない場合は切断しない (青)
        for neighbor in neighbors:
            root_canal_labels = G.nodes[neighbor].get("inheritance_root_canal_label", [])
            if root_canal_labels == [] or root_canal_labels == G.nodes[node].get("inheritance_root_canal_label", []):
                G.edges[node, neighbor]["color"] = "blue"
            else:
                G.edges[node, neighbor]["color"] = "red"
                G.nodes[neighbor]["seed"] = True

        # それ以下のエッジで赤いエッジがある場合は切断しない (緑にする)
        for neighbor in neighbors:
            if find_candidate_cut_edge(G, neighbor, direction=direction):
                G.edges[node, neighbor]["color"] = "green"
                G.nodes[neighbor]["seed"] = False

    return G


def find_candidate_cut_edge(G: nx.Graph, start_node: tuple, direction: str = "down") -> bool:
    """
    切断候補のエッジが、探索方向に存在するかどうかを確認する。
    Parameters:
        G: nx.Graph
        start_node: 探索を開始するノード
        attribute: 確認する属性 (例: "seed")
        direction: "down" or "up"
            - "down": 上から下に探索する
            - "up": 下から上に探索する
    Returns:
        True : 切断候補のエッジが存在する
        False: 切断候補のエッジが存在しない
    """
    visited_node = set() # 探索済みのノード
    stack_node = [start_node]


    while stack_node:
        node = stack_node.pop()
        # すでに探索済みの場合は何もしない
        if node in visited_node:
            continue
        visited_node.add(node) # 探索済みノードに加える

        # 隣接ノードを探索する
        for neighbor_node in G.neighbors(node):
            if direction == "down":
                # スライス番号が大きいもののみ探索する
                if neighbor_node[0] < node[0]:
                    continue
            elif direction == "up":
                # スライス番号が小さいもののみ探索する
                if neighbor_node[0] > node[0]:
                    continue

            # node = start_node の場合、隣接ノードを stack に加えて終了する
            if node == start_node:
                stack_node.append(neighbor_node)
                continue

            # 隣接ノードとのエッジの色を確認する
            edge_color = G.edges[node, neighbor_node].get("color")
            # 赤いエッジがあれば True を返す
            if edge_color == "red":
                return True
            # ノードが赤でない場合は、下の層へ進むため、 neighbor_node を stack に入れる
            stack_node.append(neighbor_node)

    # 特定の属性を持つノードが存在しないなら False を返す
    return False


def add_largest_area_seed_to_unbranched_graph(G: nx.Graph) -> list[tuple]:
    """分岐のない連結成分に、面積最大のノードを seed として追加する。"""
    graph_nodes = [
        node for node in G.nodes
        if isinstance(node[1], (int, np.integer))
    ]
    graph = G.subgraph(graph_nodes)
    added_seed_nodes = []

    for component_nodes in nx.connected_components(graph):
        component_nodes = list(component_nodes)
        if any(graph.degree[node] > 2 for node in component_nodes):
            continue
        if any(graph.nodes[node].get("seed", False) for node in component_nodes):
            continue

        seed_candidates = [
            node for node in component_nodes
            if graph.nodes[node].get("root_canal_labels", [])
            or graph.nodes[node].get("inheritance_root_canal_label", [])
        ]
        if not seed_candidates:
            continue

        target_node = max(
            seed_candidates,
            key=lambda node: int(graph.nodes[node].get("area", 0)),
        )
        G.nodes[target_node]["seed"] = True
        added_seed_nodes.append(target_node)

    return added_seed_nodes

def relabeling_labels(
    G: nx.Graph,
    volume: np.ndarray,
    root_canal_labeled_volume: np.ndarray,
    labeled_volume_per_slice: np.ndarray,
) -> np.ndarray:
    """
    seed=True のノードに対応する歯冠ラベルと、
    そのノードに対応する根管ラベルの3D領域を同じ番号で seed_volume に入れる。
    """
    label_number = 1
    seed_volume = np.zeros_like(volume, dtype=np.uint16)

    seed_nodes = [
        node
        for node, attr in G.nodes(data=True)
        if node[1] != "slice_index" and attr.get("seed", False)
    ]

    for node in seed_nodes:
        slice_index = node[0] - 1
        crown_label = node[1]

        root_canal_labels = G.nodes[node].get("root_canal_labels", [])

        # seed ノード自身に root_canal_labels がない場合は、継承ラベルを使う
        if not root_canal_labels:
            root_canal_labels = G.nodes[node].get("inheritance_root_canal_label", [])

        # 1. node に対応する歯冠側 connected component を入れる
        masked_crown_labels = labeled_volume_per_slice[slice_index] == crown_label
        seed_volume[slice_index, masked_crown_labels] = label_number

        # 2. node に対応する根管ラベルを3D全体から入れる
        for root_canal_label in root_canal_labels:
            masked_root_canal_labels = root_canal_labeled_volume == root_canal_label
            seed_volume[masked_root_canal_labels] = label_number

        label_number += 1

    return seed_volume


def find_lowest_slice_number(
        binary_volume: np.ndarray,
        smoothing_sigma: float = 2.0,
        min_distance: int = 1) -> tuple[int, np.ndarray, np.ndarray]:
    """slice 方向の面積プロファイルから最も深い谷の slice 番号を返す。

    Parameters:
        binary_volume: (slice, height, width) の歯領域マスク
        smoothing_sigma: 谷の検出に使う Gaussian 平滑化の sigma
        min_distance: 谷同士の最小間隔（slice 数）

    Returns:
        (1 始まりの slice 番号, 面積プロファイル, 平滑化プロファイル)
    """
    if binary_volume.ndim != 3:
        raise ValueError(f"binary_volume は 3 次元配列が必要です: ndim={binary_volume.ndim}")

    profile = np.count_nonzero(binary_volume, axis=(1, 2)).astype(float)
    valid_indices = np.flatnonzero(profile > 0)
    if valid_indices.size == 0:
        raise ValueError("歯領域を含む slice がありません")

    first_valid, last_valid = valid_indices[[0, -1]]
    smoothed_profile = gaussian_filter1d(profile, sigma=smoothing_sigma)
    valley_indices, _ = find_peaks(
        -smoothed_profile,
        distance=min_distance,
    )
    valley_indices = valley_indices[
        (valley_indices >= first_valid) & (valley_indices <= last_valid)
    ]

    if valley_indices.size == 0:
        valley_indices = np.array([
            first_valid + np.argmin(smoothed_profile[first_valid:last_valid + 1])
        ])

    lowest_slice_index = valley_indices[
        np.argmin(smoothed_profile[valley_indices])
    ]
    return int(lowest_slice_index + 1), profile, smoothed_profile

def create_seed_main(volume: np.ndarray, labeled_root_canal: np.ndarray, debug: bool = False):
    """
    根管ラベルをもとに、 Seed を作成する。
    Parameters:
        volume: 元の Volume データ
        labeled_root_canal: 根管ラベル付き Volume データ
    Returns:
        seed_volume: Seed Volume データ
    """
    labeled_volume_per_slice = np.zeros_like(volume, dtype=np.uint16)

    # 上から下に探索するグラフを作成する
    G_down, labeled_volume_per_slice_down = create_graph(
        volume=volume, 
        labeled_volume_root_canal=labeled_root_canal, 
        direction="down")
    G_down = edge_cut(G_down, direction="down")

    # 下から上に探索するグラフを作成する
    G_up, labeled_volume_per_slice_up = create_graph(
        volume=volume, 
        labeled_volume_root_canal=labeled_root_canal, 
        direction="up")
    G_up = edge_cut(G_up, direction="up")

    # 上下のグラフを統合する
    G = nx.compose(G_down, G_up)

    # 分岐せずに一本で続くグラフには、面積最大のノードを seed として追加する
    add_largest_area_seed_to_unbranched_graph(G)

    labeled_volume_per_slice = labeled_volume_per_slice_down + labeled_volume_per_slice_up  # 上下のグラフを統合した後も、下からのスライスラベルを使用する

    # Seed Volume を作成する
    seed_volume = relabeling_labels(
        G,
        volume,
        labeled_root_canal,
        labeled_volume_per_slice=labeled_volume_per_slice)

    if debug:
        plot_graph(G, save_path=r"./Watershed/temp/graph.png")

    return seed_volume

def main_test():
    # down の調子が悪いので確認する
    volume_path     = r"./Watershed/t-oe/20260716_NR_Label/Label_1/Label1.tif"
    labeled_root_canal_path = r"./Watershed/temp/labeled_root_canal.tif"

    volume             = load_data(volume_path)
    labeled_root_canal = load_data(labeled_root_canal_path)

    G, labeled_volume_per_slice = create_graph(volume, labeled_root_canal, direction="down")

    G = edge_cut(G, direction="down")

    plot_graph(G, save_path=r"./Watershed/temp/graph.png")

    # 赤エッジの数、 緑エッジの数、 青エッジの数を表示する
    red_edges   = [(u, v) for u, v, attr in G.edges(data=True) if attr.get("color") == "red"  ]
    green_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get("color") == "green"]
    blue_edges  = [(u, v) for u, v, attr in G.edges(data=True) if attr.get("color") == "blue" ]
    print(f"Red edges  : {len(red_edges)}")
    print(f"Green edges: {len(green_edges)}")
    print(f"Blue edges : {len(blue_edges)}")
    return

def main():
    volume_path             = r"./Watershed/t-oe/20260716_NR_Label/Label_1/Label1.tif"
    labeled_root_canal_path = r"./Watershed/t-oe/20260716_NR_Label/Label_1/CTHRs_100_Label1_labeled_root_canal.tif"

    volume             = load_data(volume_path)
    labeled_root_canal = load_data(labeled_root_canal_path)

    seed_volume = create_seed_main(volume, labeled_root_canal)

    save_volume(r"./Watershed/temp/seed_volume.tif", seed_volume, dtype=np.uint16)
    
if __name__ == "__main__":
    main()