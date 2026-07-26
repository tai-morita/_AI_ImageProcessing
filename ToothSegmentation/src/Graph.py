"""
グラフを使って Seed 探索をする
"""

from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import src

from src.DataLoad import load_data

debug = True

# 重み付きグラフの作成
def create_graph(volume: int, threshold: int, labels_connectivity: int =2):
    G = nx.Graph()
    count = 0
    for index, curr_slice in enumerate(volume):
        # スライスは1始まりにする
        slice_index = index + 1

        if debug and (slice_index >= 30 or slice_index < 12):
            continue
        count += 1

        # 表示用
        # 一番左にスライス番号を表示する
        G.add_node((slice_index, 'slice_index'), seed=False)

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
            label_area = np.sum(curr_labels == curr_label_number)
            # 面積を属性としてノード(頂点)を追加
            G.add_node((slice_index, curr_label_number), area = label_area, seed = False)
        if slice_index == 1 or count == 1:
            # 最初のスライスは直前のスライスがなくエッジが引けない
            continue

        # 直前のスライスと比較してエッジ(辺)を引く
        prev_slice_index = index-1
        prev_slice = volume[prev_slice_index]
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
            """
            edge_color = "red" # 切断しないエッジは青、切断するエッジ(Seed 候補)は赤
            # 重なっているノードが1つの場合は切断しない
            if len(overlap_label_numbers) > 1:
                for overlap_label_number in overlap_label_numbers:
                    # すべてのノードが閾値を超えている場合のみ切断候補とする
                    if threshold > G.nodes[(prev_slice_index, overlap_label_number)]["area"]:
                        # 1 つでも閾値未満のものがあれば切断しない
                        edge_color = "blue"
                        break
            else:
                edge_color = "blue"
            # 重なっているノードでエッジを引く
            for overlap_label_number in overlap_label_numbers:
                overlap_label_number = int(overlap_label_number)
                if ((slice_index, curr_label_number) in G)\
                and ((slice_index-1, overlap_label_number)) in G:
                    G.add_edge(
                        (slice_index, curr_label_number),
                        (slice_index-1, overlap_label_number),
                        color = edge_color
                    )

    return G


def find_red_edge_previous_node(G, start_node):
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
            # スライス番号が小さいもののみ探索する
            if neighbor_node[0] > node[0]:
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
            if (find_red_edge_previous_node(G, previous_node)):
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
def plot_graph(G: nx):
    pos = {}
    for node in G.nodes:
        slice_index, label = node
        # スライス番号を左端に表示する
        if label == 'slice_index':
            pos[node] = (0, slice_index)
        else:
            pos[node] = (label, slice_index) # x:スライス順, y:ラベル番号で上下にずらす

    plt.figure(figsize=(12, 20))

    # 描画するものを決める
    labels = {}
    for node in G.nodes:
        # スライス番号表示用のノードとメインのノードで表示するものを分ける
        if node[1] == 'slice_index':
            labels[node] = f"slice {node[0]}"
        else:
            # メインは面積
            labels[node] = f"{G.nodes[node]['area']}"
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

    plt.show()

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

if __name__ == "__main__":
    
    input_path = r"C:\Users\morit\Desktop\work\study\Watershed\Data\Input\20260313_test\label_map_1_annotate.tif"
    test_connected_component(input_path)
    # volume = load_data(input_path)
    # G = create_graph(volume, 500)
    # find_previous_red_edge(G)
    # plot_graph(G)