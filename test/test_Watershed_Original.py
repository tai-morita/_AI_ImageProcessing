import numpy as np
import heapq
import csv
import tifffile
from skimage import io, color, exposure, filters, morphology, util
from . test_set_array import generate_height_map_5basins

# 自作のwatershed実装
"""
2d
アルゴリズム
引数: 2d image, seed(分類したいオブジェクトの極小に近い点)
1. 前景、背景を二値化する
2. 前景の距離変換を行う
3. 距離の小さい順に以下の処理を行う 距離=d, 初回のみ0を実行, 以降は1からループ
    0) seed点をラベリングする: coords=[height, width, label]とした配列にする
    while q:
        for d
            1) 距離dの点を全てキューに入れる
            2) qから点pを取り出す
            3) pの近傍を巡回(8近傍)
            4) それぞれの状態に応じて、以下の処理を行う
                if 近傍が既ラベル1種のみ
                    pをそのラベルにする
                elif 近傍が複数の既ラベル or WSHED
                    pをWSHEDにする
                elif 近傍が未処理のMASKのみ
                    pを再度キューに入れる(次の距離層で処理するため)
"""

def neighbors_lin(coords, arr_shape, neibors=8):
    # 8近傍の座標を返す
    # ただし、画像の範囲外の座標は返さない (1ライン増やすのとどっちがのであろう)
    height = coords[0]
    width = coords[1]
    if neibors == 4:
        coords_neibor = [(height-1,  width), (height+1, width), 
                (height, width-1), (height, width+1)]
    elif neibors == 8:
        coords_neibor = [(height-1, width-1), (height-1, width), 
                (height-1, width+1), (height, width-1), 
                (height, width+1), (height+1, width-1), 
                (height+1, width), (height+1, width+1)]
    # 画像の範囲外の座標は返さない
    coords_neibor = [coord for coord in coords_neibor if 0 <= coord[0] < arr_shape[0] and 0 <= coord[1] < arr_shape[1]]
    return coords_neibor

def count_label(coords, arr2d_label):
    # 座標のリスト内にラベルが何種類あるか調べる
    label_count = set()
    for coord in coords:
        label_count.add(arr2d_label[coord])
    # 0はMASKを表すので、0は除外する
    # -1も除外する
    label_count.discard(0)
    label_count.discard(-1)
    return label_count

def push_neighbors_to_queue(coords, arr2d_label, arr2d_height, list_queue):
    # 近傍かつミラベルの点を優先度付きキューに入れる
    for coord in coords:
        if arr2d_label[coord] == 0: # MASKのみキューに入れる
            heapq.heappush(list_queue, (arr2d_height[coord], coord))

def argorism_main(array2d, seed_coords):
    list_queue = [] # (height, width)の配列
    arr2d_label = np.zeros_like(array2d) # ラベリング結果を格納する配列
    # seed点をラベリングする。近傍にseedn点が複数ある場合は、同一のラベルにする
    value_label = 1
    for seed in seed_coords:
        arr2d_label[seed] = value_label
        value_label += 1
    # ラベリングされた点(最初なのでseed)の近傍を優先度付きキューに入れる
    for seed in seed_coords:
        coords_neighbors = neighbors_lin(seed, arr2d_label.shape, neibors=8)
        push_neighbors_to_queue(coords_neighbors, arr2d_label, array2d, list_queue)

    while list_queue:
        coord_p = heapq.heappop(list_queue)[1]
        if True:
            # テスト用: seed点は処理しない
            if arr2d_label[coord_p] != 0:
                continue
        # 近傍を巡回
        coords_neighbors = neighbors_lin(coord_p, arr2d_label.shape, neibors=8)
        value_label_count = count_label(coords_neighbors, arr2d_label)
        if value_label_count == set():
            # 近傍が未処理のMASKのみ -> pを再度キューに入れる(次の距離層で処理するため)
            heapq.heappush(list_queue, (array2d[coord_p], coord_p))
        elif len(value_label_count) == 1:
            # 近傍が既ラベル1種のみ -> pをそのラベルにする
            arr2d_label[coord_p] = value_label_count.pop()
            # ラベリングしたらその点の近傍をキューに入れる
            push_neighbors_to_queue(coords_neighbors, arr2d_label, array2d, list_queue)
        elif len(value_label_count) > 1:
            # 近傍が複数の既ラベル or WSHED -> pをWSHEDにする
            arr2d_label[coord_p] = -1 # WSHEDは-1で表す
    return arr2d_label
            
def origianl_watershed_2d(arr2d, seed_coords): 
    # アルゴリズムを実行する
    array2d_label = argorism_main(np.array(arr2d), seed_coords)
    return array2d_label

def get_seed_coords(img):
    # 画像からseed点の座標を取得する
    # グレースケール画像
    arr2d_img = io.imread(img, as_gray=True)
    seed_coords = np.argwhere(arr2d_img >= 0.5) 
    return seed_coords

if __name__ == "__main__":
    image_original = io.imread('./study/test/Input/coin_gray.png', as_gray=True)
    image_seed = r"./study/test/Input/seed_coin.png"
    seed_coords = get_seed_coords(image_seed) # seed点の座標を取得する
    array2d_label = origianl_watershed_2d(image_original, seed_coords)
    print(array2d_label)

    """
    # CSVに出力
    np.savetxt('./study/test/Output/watershed_result.csv', array2d_label, delimiter=',', fmt='%d')
    print("CSVに出力しました: ./study/test/Output/watershed_result.csv")
    """
    # 画像に出力
    # ラベルを色に変換する
    # ラベル0は背景、-1はWSHEDとする
    array2d_label_color = np.zeros((array2d_label.shape[0], array2d_label.shape[1], 3), dtype=np.uint8)
    for i in range(array2d_label.shape[0]):
        for j in range(array2d_label.shape[1]):
            if array2d_label[i, j] == 0:
                array2d_label_color[i, j] = [0, 0, 0] # 背景は黒
            elif array2d_label[i, j] == -1:
                array2d_label_color[i, j] = [255, 255, 255] # WSHEDは白
            else:
                # ラベルに応じて色を変える (今回は適当に色を割り当てる)
                array2d_label_color[i, j] = [(array2d_label[i, j] * 50) % 256, (array2d_label[i, j] * 80) % 256, (array2d_label[i, j] * 110) % 256]
    io.imsave('./study/test/Output/watershed_result.png', array2d_label_color)
    print("画像に出力しました: ./study/test/Output/watershed_result.png")