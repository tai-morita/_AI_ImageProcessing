"""
1. グラフを作る
    1-0. volume の各スライスをラベル付けする
    1-1. ノードを作る
        属性: 
            - slice: スライス番号
            - label: ラベル番号 (Connected Component のラベル)
            - root_canal_label: 根管ラベル番号 (input)
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

from .DataLoad import load_data, save_volume
from .LabeledRootCanal import labeled_for_root_canal

def create_graph(volume: np.ndarray, 
                 labeled_volume_rootcanal: np.ndarray, 
                 labels_connectivity: int =2, 
                 direction: str = "down") -> nx.Graph:
    G = nx.Graph()
    count = 0

    for index, curr_slice in enumerate(volume):
        # スライスは1始まりにする
        slice_index = index + 1

        print(f"Processing slice {index + 1}/{volume.shape[0]}")
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
                    prev_root_canal_labels = G.nodes[(prev_slice_index, overlap_label_number)]["root_canal"]
                    root_canal_labels = G.nodes[(slice_index, curr_label_number)]["root_canal"]
                    root_canal_labels.extend(prev_root_canal_labels)
                    G.nodes[(slice_index, curr_label_number)]["root_canal"] = list(dict.fromkeys(root_canal_labels))
        # 表示用
        # 一番左にスライス番号を表示する
        G.add_node((slice_index, 'slice_index'), seed=False)

    return G


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
            labels[node] = f"{G.nodes[node]['area']}\n{G.nodes[node]['root_canal']}"
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

if __name__ == "__main__":
    volume_path     = r"./study/Watershed/t-oe/20260716_NR_Label/test/Label1/Label1.tif"
    root_canal_path = r"./study/Watershed/t-oe/20260716_NR_Label/test/Label1/CTHRs_100_Label1_ternary.tif"

    volume     = load_data(volume_path)
    root_canal = load_data(root_canal_path)

    labeled_root_canal = labeled_for_root_canal(volume=root_canal, target_value=1)
    save_volume(r"./study/Watershed/temp/labeled_root_canal.tif", labeled_root_canal, dtype=np.uint16)

    G = create_graph(volume, labeled_root_canal)

    plot_graph(G, save_path=r"./study/Watershed/temp/graph.png")
