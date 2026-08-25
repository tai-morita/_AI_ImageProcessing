"""
グラフを使って Seed 探索をする
"""

from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .DataLoad import load_data, save_volume
from .LabeledRootCanal import labeled_for_root_canal


debug = False

# 重み付きグラフの作成
"""
ノードは各スライスの Connected component とする。
各ノードは (slice_index, label_number) というタプルで表される。
    - label_number は各スライスでの 1 から始まる連番で、0 は背景を表す。
各ノードには以下の属性が付与されている。
    - area: 面積
    - seed: Seed かどうかのフラグ (True/False)
    - root_canal: 根管のラベル (True ならその番号, False なら 0)
エッジは隣接するスライス間の Connected component の重なりに基づいて引かれる。
エッジには以下の属性が付与されている。
    - color: エッジの色 (赤: 切断候補, 青: 切断しない, 緑: 切断候補だが下の層に赤エッジがあるので切断しない)

"""
def create_graph(volume: np.ndarray, labeled_volume_rootcanal: np.ndarray, labels_connectivity: int =2, direction: str = "down") -> nx.Graph:
    G = nx.Graph()
    count = 0
    edge_cut = True
    for index, curr_slice in enumerate(volume):
        # スライスは1始まりにする
        slice_index = index + 1
        if debug and (slice_index >= 400 or slice_index < 0):
            continue

        print(f"Processing slice {index + 1}/{volume.shape[0]}")
        count += 1

        """
        rank 次元の連結成分とみなす bool 値が入る
        rank=2, connectivity=1 なら 4 近傍なので
        False True False
        True  True True
        False True False
        """

        structure = ndi.generate_binary_structure(rank=2, connectivity=labels_connectivity)
        # structure 近傍を隣接画素としてラベル付けをする
        curr_labels, curr_label_numbers = ndi.label(curr_slice, structure=structure)

        for curr_label_number in range(1, curr_label_numbers+1):
            # 面積が属性となるため、ラベルごとに面積を計算する
            component_mask = (curr_labels == curr_label_number)
            label_area = np.sum(component_mask)

            # 根管のラベルを取得する
            root_canal_labels = get_root_canal_labels_for_component(
                component_mask=component_mask,
                labeled_volume_rootcanal=labeled_volume_rootcanal[index],
            )

            # 面積, 根管のラベルを属性としてノード(頂点)を追加
            G.add_node(
                (slice_index, curr_label_number),
                area=label_area,
                seed=False,
                root_canal=root_canal_labels
            )
        if slice_index == 1 or count == 1:
            # 最初のスライスは直前のスライスがなくエッジが引けない
            continue

        # 直前のスライスと比較してエッジ(辺)を引く
        prev_array_index = index-1
        prev_slice_index = slice_index- 1
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

            # スライス間で重なっているノードにはエッジを引く
            """
            次の条件を満たす場合はエッジ切断候補 (切断候補の直前のノードを Seed とする)
            1. 重なっているノードが 2 つ以上ある
            2. 重なっているノードの面積がすべて閾値以上 (細かいノイズは除去されている前提?)
            X. 重なるノード A, B で異なる根管ラベルを保持している
            """
            edge_color = "blue" # 切断しないエッジは青、切断するエッジ(Seed 候補)は赤
            # 重なっているノードが1つの場合は切断しない
            if len(overlap_label_numbers) > 1:
                """
                root_canal_labels_set = set()
                for overlap_label_number in overlap_label_numbers:
                    # 各ノードの根管ラベルを取得する
                    prev_root_canal_labels = G.nodes[(prev_slice_index, overlap_label_number)]["root_canal"]
                    root_canal_labels_set.update(prev_root_canal_labels)
                    # 根管ラベルが異なる場合は切断候補とする
                    if len(root_canal_labels_set) > 1:
                        edge_color = "red"
                        break
                """
                edge_color = "red"
            # 重なっているノードでエッジを引く
            for overlap_label_number in overlap_label_numbers:
                overlap_label_number = int(overlap_label_number)
                if ((slice_index, curr_label_number) in G)\
                and ((prev_slice_index, overlap_label_number)) in G:
                    G.add_edge(
                        (slice_index, curr_label_number),
                        (prev_slice_index, overlap_label_number),
                        color = edge_color
                    )
                    # 根管ラベルは、前スライスの根管ラベルを追加する
                    prev_root_canal_labels = G.nodes[(prev_slice_index, overlap_label_number)]["root_canal"]
                    root_canal_labels = G.nodes[(slice_index, curr_label_number)]["root_canal"]
                    root_canal_labels.extend(prev_root_canal_labels)
                    G.nodes[(slice_index, curr_label_number)]["root_canal"] = list(dict.fromkeys(root_canal_labels))
        # 表示用
        # 一番左にスライス番号を表示する
        G.add_node((slice_index, 'slice_index'), seed=False)

    return G

def edge_cut(G: nx.Graph, direction: str = "down"):
    """
    グラフのエッジを切断する。
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
            - "down": 上から下に探索する
            - "up": 下から上に探索する
    """
    for node1, node2, attr in G.edges(data=True):
        # スライス番号を取得し、前後スライスのノードであることを確認する
        slice_index1 = G.nodes[node1].get("slice_index", node1[0])
        slice_index2 = G.nodes[node2].get("slice_index", node2[0])
        if slice_index1 == slice_index2:
            # 同じスライスのノード同士は無視する (ここに来ないはず)
            continue
        if slice_index1 > slice_index2:
            prev_node = node2
            curr_node = node1
        else:
            prev_node = node1
            curr_node = node2
        
        if attr.get("color") == "red":
            # 切断候補のエッジを確認する
            # 前後スライスのノードの根管ラベルを取得する
            prev_node_root_canal = G.nodes[prev_node]["root_canal"]
            curr_node_root_canal = G.nodes[curr_node]["root_canal"]
            # 根管ラベルを比較し、同じ場合は切断しない (緑にする)
            if set(prev_node_root_canal) == set(curr_node_root_canal):
                G.edges[node1, node2]["color"] = "green"
                continue
            # 根管ラベルが異なる場合は切断したい
            # 下の層に赤いエッジがあるか確認する
            if find_red_edge_previous_node(G, prev_node, direction=direction):
                # 赤いエッジがある場合は切断しない (緑にする)
                G.edges[node1, node2]["color"] = "green"
                continue



def get_root_canal_labels_for_component(component_mask: np.ndarray, labeled_volume_rootcanal: np.ndarray) -> list[int]:
    """
    component_mask: ある connected component の True/False のマスク画像 (2D)
    labeled_volume_rootcanal: root canal のラベル画像 (背景は 0)

    Returns:
        component_mask と重なっている root canal ラベルの一覧
        例: [3] or [3, 7] or []
    """
    # component_mask と labeled_volume_rootcanal の形状が一致しているか確認する
    if component_mask.shape != labeled_volume_rootcanal.shape:
        raise ValueError(f"component_mask と labeled_volume_rootcanal の形状が一致しません: component_mask.shape={component_mask.shape}, labeled_volume_rootcanal.shape={labeled_volume_rootcanal.shape}")
    # 成分と重なる根管ラベルがなければ、空のリストを返す
    overlapped = np.unique(labeled_volume_rootcanal[component_mask])
    overlapped = overlapped[overlapped != 0]

    return overlapped.astype(int).tolist()

def find_red_edge_previous_node(G, start_node, direction: str = "down") -> bool:
    # start_node 以下のスライスをエッジをたどって探索していき、赤ノードがないかを探す
    # return true: 赤エッジない, false: 赤エッジある
    visited_node = set() # 探索済みのノード
    stack_node = [start_node]

    while stack_node:
        node = stack_node.pop()
        # すでに探索済みの場合は何もしない
        if node in visited_node:
            continue
        visited_node.add(node) # 探索済みノードに加える

        # 隣接したノードを探索する
        for neighbor_node in G.neighbors(node):
            if direction == "down":
                # スライス番号が小さいもののみ探索する
                if neighbor_node[0] > node[0]:
                    continue
            elif direction == "up":
                # スライス番号が大きいもののみ探索する
                if neighbor_node[0] < node[0]:
                    continue
            # エッジが赤なら終了
            if G[node][neighbor_node].get("color") == "red":
                return True
            # エッジが赤でない場合は、下の層へ進むため、 neighbor_node を stack に入れる
            stack_node.append(neighbor_node)

    # 赤エッジが存在しないならfalse: 切断しない
    return False

def find_previous_red_edge(G: nx):
    # 赤いエッジでそれ以下のsliceに赤エッジがあれば緑にする
    count = 0
    # エッジから (Slice-1 のノード, Slice のノード, 属性)を取り出す
    for previous_node, current_node, attr in \
        sorted(G.edges(data=True),\
                key=lambda edge: max(edge[0][0], edge[1][0]),\
                reverse=True):
        if attr.get("color") == "red":
            # 赤いエッジが見つかった
            if (find_red_edge_previous_node(G, previous_node, direction="down")):
                # それ以下のエッジで赤いものがあったので、緑に変更する
                G.edges[current_node, previous_node]["color"] = "green"
                continue
            else:
                # 赤いエッジがない
                # 属性に Seed を追加する
                # True のノードを Seed にするようにしている
                G.nodes[previous_node]["seed"] = True
                continue

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
            labels[node] = f"{G.nodes[node]['area']}: {G.nodes[node]['root_canal']}"
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


def test_connected_component(input_path):
    volume = load_data(input_path)
    volume = volume[0: 50]
    slice_indices = []
    areas = []

    for index, curr_slice in enumerate(volume):
        labels, label_numbers = ndi.label(curr_slice)

        for label_number in range(1, label_numbers + 1):
            label_area = np.sum(labels == label_number)

            slice_indices.append(index + 1)
            areas.append(label_area)

    plt.scatter(slice_indices, areas)

    plt.ylim(0, 4000)
    plt.xlabel("Slice index")
    plt.ylabel("Connected component area")
    plt.title("Connected component area by slice")

    plt.show()
    if True:
        plt.savefig(r"./study/Watershed/temp/connected_component_area_by_slice.png", dpi=300)


if __name__ == "__main__":
    
    volume_path = r"./study/Watershed/t-oe/20260716_NR_Label/test/Label1/Label1.tif"
    root_canal_path = r"./study/Watershed/t-oe/20260716_NR_Label/test/Label1/CTHRs_100_Label1_ternary.tif"
    # root canal のラベルをツクル
    volume = load_data(volume_path)
    root_canal = load_data(root_canal_path)
    labeled_root_canal = labeled_for_root_canal(volume=root_canal, target_value=1)
    G = create_graph(volume, labeled_root_canal)
    # find_previous_red_edge(G)
    # edge_cut(G, direction="up")
    plot_graph(G, save_path=r"./study/Watershed/temp/graph.png")

    from collections import Counter
    edge_color_counts = Counter(
        attributes.get("color", "black")
        for _, _, attributes in G.edges(data=True)
    )
    print(f"赤エッジ: {edge_color_counts['red']}")
    print(f"緑エッジ: {edge_color_counts['green']}")
    """
    """
    